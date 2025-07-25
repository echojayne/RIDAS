import os
import json
import math
import re
import logging
import sys
import sqlite3
import pandas as pd
import csv
from fractions import Fraction
from openai import OpenAI
from tqdm import tqdm

from prompts import PROMPTS

# ===== Logging setup with colored output =====
GREEN = "\033[32m"
RED = "\033[31m"
RESET = "\033[0m"

class ColoredFormatter(logging.Formatter):
    """
    Logging Formatter to add colors based on log level.
    """
    FORMATS = {
        logging.INFO:    GREEN + "%(asctime)s - " + "%(levelname)s" + RESET + " - %(message)s",
        logging.WARNING: RED   + "%(asctime)s - " + "%(levelname)s" + RESET + " - %(message)s",
        logging.ERROR:   RED   + "%(asctime)s - " + "%(levelname)s" + RESET + " - %(message)s",
        logging.CRITICAL:RED   + "%(asctime)s - " + "%(levelname)s" + RESET + " - %(message)s",
        logging.DEBUG:   GREEN + "%(asctime)s - " + "%(levelname)s" + RESET + " - %(message)s",
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt, "%Y-%m-%d %H:%M:%S")
        return formatter.format(record)

# Configure root logger
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(ColoredFormatter())
logger = logging.getLogger("ida_logger")
logger.addHandler(handler)
logger.setLevel(logging.INFO)

# ===== Configuration loading =====
ROOT_PATH = os.path.dirname(__file__)
CFG_PATH = os.path.join(ROOT_PATH, "config.json")
with open(CFG_PATH, "r") as f:
    config = json.load(f)

def extract_config(text: str) -> dict | None:
    pattern = r"```json\s*(.*?)\s*```"
    match = re.search(pattern, text, re.DOTALL)
    if not match:
        logger.error("Can not find ```json ``` code block in the text.")
        return None
    json_string = match.group(1).strip()

    bare_expr_pattern = re.compile(r'("[\w_]+"\s*:\s*)([\d\.\s\(\)\+\-\*\/]+(?:[\s\*\/\+\-][\d\s\.\(\)\+\-\*\/]+)+)')
    while True:
        bare_match = bare_expr_pattern.search(json_string)
        if not bare_match:
            break

        key_part = bare_match.group(1)
        expression_str = bare_match.group(2).strip()
        
        if not re.fullmatch(r'[\d\s\.\+\-\*\/\(\)]+', expression_str):
            logger.warning(f"Finding a bare expression but contains unsafe characters, skipped: '{expression_str}'")
            json_string = json_string.replace(bare_match.group(0), f'{key_part}null', 1)
            continue
        
        try:
            evaluated_value = eval(expression_str)
            replacement_str = f'{key_part}{json.dumps(evaluated_value)}'
            json_string = json_string.replace(bare_match.group(0), replacement_str, 1)
            # logger.info(f"Successfully evaluated bare expression '{expression_str}' to {evaluated_value}.")
        except Exception as e:
            logger.error(f"Computing bare expression '{expression_str}' failed: {e}. Replacing with null.")
            json_string = json_string.replace(bare_match.group(0), f'{key_part}null', 1)

    str_expr_pattern = re.compile(r'("[\w_]+"\s*:\s*)"([\d\s\.\+\-\*\/\(\)]+(?:[\s\*\/\+\-][\d\s\.\(\)\+\-\*\/]+)+)"')
    while True:
        str_match = str_expr_pattern.search(json_string)
        if not str_match:
            break

        key_part = str_match.group(1)
        expression_str = str_match.group(2)

        try:
            evaluated_value = eval(expression_str)
            replacement_str = f'{key_part}{json.dumps(evaluated_value)}'
            json_string = json_string.replace(str_match.group(0), replacement_str, 1)
            logger.info(f"Successfully evaluated string expression '\"{expression_str}\"' to {evaluated_value}.")
        except Exception as e:
            logger.error(f"Computing string expression '\"{expression_str}\"' failed: {e}. Replacing with null.")
            json_string = json_string.replace(str_match.group(0), f'{key_part}null', 1)

    try:
        data = json.loads(json_string)
    except json.JSONDecodeError as e:
        logger.error(f"JSON block still invalid after processing: {e}")
        logger.debug(f"Original JSON string: {json_string}")
        return None

    def _recursive_type_correction(obj):
        if isinstance(obj, dict):
            return {k: _recursive_type_correction(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_recursive_type_correction(elem) for elem in obj]
        if isinstance(obj, str):
            if obj.isdigit():
                return int(obj)
            try:
                if obj.count('.') == 1 and all(part.isdigit() for part in obj.split('.')):
                     return float(obj)
            except (ValueError, TypeError):
                pass
        return obj

    corrected_data = _recursive_type_correction(data)
    return corrected_data

def database_init(path, db_file="Tables/Memory/Memory.db"):
    """
    Initializes the database by reading all CSV files from a directory
    and storing them as tables in an SQLite database.
    """
    entries = os.listdir(path)
    files = [f for f in entries if os.path.isfile(os.path.join(path, f)) and f.endswith(".csv")]
    
    for file in files:
        df = pd.read_csv(os.path.join(path, file))
        conn = sqlite3.connect(db_file)
        # Use filename without extension as table name
        table_name = os.path.splitext(file)[0]
        df.to_sql(table_name, conn, if_exists="replace", index=False)
        conn.close()
        logger.info(f"Loaded {file} into table '{table_name}' in {db_file}")

def sql_execute(sql, db_file="experience_configuration.db"):
    """
    Executes an SQL statement and returns the result.
    """
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    cursor.execute(sql)
    result = cursor.fetchall()
    conn.commit()
    conn.close()
    return result

class SystemConfig:
    """
    Manages the overall system configuration and state.
    """
    def __init__(self, client, llm_config, ida_config):
        self.max_user_num = ida_config.get("max_user_num", 1000)
        self.total_bandwidth = ida_config.get("total_bandwidth", 100)  # MHz
        self.transmission_time = ida_config.get("transmission_time", 5e-4) # s
        self.code_rate_options = ida_config.get("code_rate_options", [1/2, 3/5, 2/3, 3/4, 4/5, 5/6, 8/9, 9/10])
        
        self.current_user_num = 0
        self.used_bandwidth = 0  # MHz

        self.db_file = ida_config["database_file"]
        self.experience_table_name = "experience_configuration"
        self.user_records_table = "user_record"

        self.llm = client
        self.llm_config = llm_config
        
        self.reset() # Ensure a clean state on initialization

    def get_current_user_num(self) -> int:
        return self.current_user_num

    def get_used_bandwidth(self) -> float:
        return self.used_bandwidth

    def get_total_bandwidth(self) -> float:
        return self.total_bandwidth

    def compute_bandwidth(self, bitstream_length, snr, code_rate, max_transmission_latency) -> float:
        """
        Computes the required bandwidth in MHz based on the Shannon-Hartley theorem.
        """
        # C = B * log2(1 + SNR) => B = C / log2(1 + SNR)
        # C = Data / Time = (bitstream_length / code_rate) / max_transmission_latency
        channel_capacity = (bitstream_length / code_rate) / max_transmission_latency
        snr_linear = 10 ** (snr / 10)
        bandwidth_hz = channel_capacity / math.log(1 + snr_linear, 2)
        return bandwidth_hz / 1e6  # Convert Hz to MHz

    def ida(self, message: list) -> str:
        """
        Interacts with the Large Language Model to get a decision.
        """
        params = {
            "model": self.llm_config.get("model"),
            "temperature": self.llm_config.get("temperature", 0.7),
            "top_p": self.llm_config.get("top_p", 1.0),
            "messages": message
        }
        response = self.llm.chat.completions.create(**params)
        return response.choices[0].message.content.strip()

    def _sql_execute(self, sql: str) -> list:
        """
        Executes an SQL statement on the system's database.
        """
        conn = sqlite3.connect(self.db_file)
        cursor = conn.cursor()
        cursor.execute(sql)
        result = cursor.fetchall()
        conn.commit()
        conn.close()
        return result

    def get_experience_configuration(self) -> list[dict]:
        conn = sqlite3.connect(self.db_file)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        try:
            cur.execute(f"SELECT * FROM {self.experience_table_name}")
            rows = cur.fetchall()
            return [dict(row) for row in rows]
        finally:
            conn.close()

    def get_user_records(self) -> list[dict]:
        conn = sqlite3.connect(self.db_file)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        try:
            cur.execute(f"SELECT * FROM {self.user_records_table}")
            rows = cur.fetchall()
            return [dict(row) for row in rows]
        finally:
            conn.close()

    def bandwidth_shrinkage(self, user_config: dict, required_bandwidth: float) -> bool:
        """
        Attempts to reduce overall bandwidth usage by adjusting existing user configurations.
        """
        modified_config = None
        message = PROMPTS.bandwidth_adjustment(user_config, self, required_bandwidth)
        while modified_config is None:
            response = self.ida(message)
            modified_config = extract_config(response)
        can_be_adjusted = modified_config.get("can_be_adjusted", False)
        if not can_be_adjusted:
            return False

        adjusted_successfully = False
        for mod_user_key in modified_config.keys():
            if mod_user_key[0:-1] == "user":
                mod_user = modified_config[mod_user_key]
                logger.info(f"LLM suggests adjusting config for user_id: {mod_user['user_id']}")
                new_user_config = {}
                new_user_config["user_id"] = mod_user["user_id"]
                # Fetch existing user data
                new_user_config["snr"] = float(self._sql_execute(f"SELECT snr FROM {self.user_records_table} WHERE user_id = {mod_user['user_id']}")[0][0])
                new_user_config["max_transmission_latency"] = float(self._sql_execute(f"SELECT max_transmission_latency FROM {self.user_records_table} WHERE user_id = {mod_user['user_id']}")[0][0])
                # Apply modifications
                for subkey, subvalue in mod_user["modified_configuration"].items():
                    new_user_config[subkey] = subvalue
                adjusted_successfully = self.allocated_resource_adjustment(new_user_config)
        return adjusted_successfully

    def add_user(self, user_config: dict) -> bool:
        """
        Adds a new user to the system, adjusting bandwidth if necessary.
        """
        attempt = 0
        can_be_adjusted = True
        
        # Loop while required bandwidth exceeds available, and adjustments are possible
        while user_config["available_bandwidth"] + self.used_bandwidth > self.total_bandwidth and can_be_adjusted and attempt < 10:
            attempt += 1
            logger.warning("="*80)
            logger.warning(f"Bandwidth adjustment attempt #{attempt}:")
            logger.warning(f"Current usage: {self.used_bandwidth:.4f} MHz. Adding user requires: {user_config['available_bandwidth']:.4f} MHz. Total would be: {self.used_bandwidth + user_config['available_bandwidth']:.4f} MHz")
            
            try:
                can_be_adjusted = self.bandwidth_shrinkage(user_config, user_config["available_bandwidth"])
            except Exception as e:
                logger.error(f"Bandwidth adjustment failed with an exception: {e}")
                return False

        # If bandwidth is still insufficient after adjustments, reject user
        if user_config["available_bandwidth"] + self.used_bandwidth > self.total_bandwidth and not can_be_adjusted:
            logger.error("Bandwidth adjustment failed. No free bandwidth and no users could be adjusted.")
            return False

        self.used_bandwidth += user_config["available_bandwidth"]
        
        add_user_sql = f"""
        INSERT INTO {self.user_records_table} (user_id, snr, code_rate, max_transmission_latency, acceptance_of_distortion, quantify_bits, decomposition_rank, experience_bitstream_length, available_bandwidth)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
        """
        params = (
            user_config["user_id"], user_config["snr"], user_config["code_rate"],
            user_config["max_transmission_latency"], user_config["acceptance_of_distortion"],
            user_config["quantify_bits"], user_config["decomposition_rank"],
            user_config["experience_bitstream_length"], user_config["available_bandwidth"]
        )
        
        # Use parameterized query to prevent SQL injection
        conn = sqlite3.connect(self.db_file)
        cursor = conn.cursor()
        cursor.execute(add_user_sql, params)
        conn.commit()
        conn.close()

        num = self.current_user_num + 1
        if 0 <= num <= self.max_user_num:
            self.current_user_num = num
        else:
            logger.error(f"User count must be between 0 and {self.max_user_num}")
            return False
            
        logger.info(f"Successfully added user {user_config['user_id']}.")
        return True

    def allocated_resource_adjustment(self, user_config: dict) -> bool:
        """
        Applies a modified configuration to an existing user and updates the database.
        """
        bitstream_length_result = self._sql_execute(f"SELECT bitstream_length FROM {self.experience_table_name} WHERE quantify_bits = {user_config['quantify_bits']} AND decomposition_rank = {user_config['decomposition_rank']}")
        if not bitstream_length_result:
            logger.error(f"No experience configuration found for bits={user_config['quantify_bits']}, rank={user_config['decomposition_rank']}")
            return False
        experience_bitstream_length = bitstream_length_result[0][0]
        new_bandwidth = self.compute_bandwidth(
            experience_bitstream_length,
            user_config["snr"],
            user_config["code_rate"],
            user_config["max_transmission_latency"]
        )
        original_bandwidth = float(self._sql_execute(f"SELECT available_bandwidth FROM {self.user_records_table} WHERE user_id = {user_config['user_id']}")[0][0])
        if new_bandwidth < original_bandwidth:
            update_sql = f"""
            UPDATE {self.user_records_table}
            SET code_rate = {user_config["code_rate"]}, quantify_bits = {user_config["quantify_bits"]}, decomposition_rank = {user_config["decomposition_rank"]},
                experience_bitstream_length = {experience_bitstream_length}, available_bandwidth = {new_bandwidth}
            WHERE user_id = {user_config["user_id"]}
            """
            self._sql_execute(update_sql)
            # Recalculate total used bandwidth from the source of truth (the database)
            original_used_bandwidth = self.used_bandwidth
            self.used_bandwidth = self._sql_execute(f"SELECT SUM(available_bandwidth) FROM {self.user_records_table}")[0][0] or 0
            logger.info(f"User {user_config['user_id']} config adjusted. Original bandwidth: {original_bandwidth:.4f} MHz. New bandwidth: {new_bandwidth:.4f} MHz.")
            logger.info(f"Total used bandwidth updated from {original_used_bandwidth:.4f} MHz to {self.used_bandwidth:.4f} MHz.")
            return True
        else:
            logger.warning(f"Adjustment for user {user_config['user_id']} failed. New bandwidth {new_bandwidth:.4f} is not less than original {original_bandwidth:.4f} MHz.")
            return False

    def reset(self):
        """Resets the system state, clearing all user records."""
        logger.info("Resetting system state.")
        self.current_user_num = 0
        self.used_bandwidth = 0.0
        self._sql_execute(f"DELETE FROM {self.user_records_table}")
        # self._sql_execute(f"UPDATE sqlite_sequence SET seq = 0 WHERE name = '{self.user_records_table}'")

def export_table_to_csv(table_name: str, csv_path: str, db_file: str):
    """
    Exports a specified SQLite table to a CSV file.
    """
    pragma_sql = f"PRAGMA table_info({table_name});"
    info = sql_execute(pragma_sql, db_file)
    columns = [col[1] for col in info]

    select_sql = f"SELECT * FROM {table_name};"
    rows = sql_execute(select_sql, db_file)

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        writer.writerows(rows)
    logger.info(f"Exported {len(rows)} records from table '{table_name}' to '{csv_path}'.")

def user_add_process(system_config: SystemConfig, user_config: dict) -> bool:
    """
    Manages the full process of adding a single user, from LLM consultation to system state update.
    """
    message = PROMPTS.bandwidth_allocation(user_config, system_config)
    with open("Prompt.txt", 'a', encoding='utf-8') as f:
        f.write(message[-1]['content'])
        f.write('\n' + '=' * 80 + '\n')
    user_end_config = None
    
    logger.info(f"Consulting LLM for initial configuration for user {user_config['user_id']}...")
    while user_end_config is None:
        action = system_config.ida(message)
        with open("Action.txt", 'a', encoding='utf-8') as f:
            f.write(action)
            f.write('\n' + '=' * 80 + '\n')
        user_end_config = extract_config(action)

    bitstream_length_result = sql_execute(f"SELECT bitstream_length FROM {system_config.experience_table_name} WHERE quantify_bits = {user_end_config['quantify_bits']} AND decomposition_rank = {user_end_config['decomposition_rank']}", system_config.db_file)
    if not bitstream_length_result:
        logger.error(f"LLM proposed an invalid configuration (bits={user_end_config['quantify_bits']}, rank={user_end_config['decomposition_rank']}) which does not exist in the experience table.")
        return False

    user_config["experience_bitstream_length"] = float(bitstream_length_result[0][0])
    user_end_config["code_rate"] = float(Fraction(user_end_config["code_rate"]))
    
    user_config["available_bandwidth"] = system_config.compute_bandwidth(
        user_config["experience_bitstream_length"],
        user_config["snr"],
        user_end_config["code_rate"],
        user_config["max_transmission_latency"]
    )

    # Update original user_config with LLM's decisions
    for key, value in user_end_config.items():
        user_config[key] = value

    return system_config.add_user(user_config)


if __name__ == "__main__":
    logger.info("Initializing database from CSV files...")
    database_init("Tables/Memory", config["IDA"]["database_file"])

    llm_config = config["IDA"]["LLM"]
    client = OpenAI(
        api_key=llm_config["API_KEY"],
        base_url=llm_config["base_url"],
    )

    logger.info("Initializing system configuration...")
    system_config = SystemConfig(client, llm_config, config["IDA"])

    with open(config["IDA"]["user_queue"], "r", encoding="utf-8") as f:
        initial_user_configs = json.load(f)
    
    logger.info(f"Starting to process {len(initial_user_configs)} users from the queue...")
    
    for idx, user_config in enumerate(initial_user_configs):
        logger.info("-" * 80)
        logger.info(f"Processing user #{idx+1} (ID: {user_config['user_id']})...")
        logger.info(f"SNR: {user_config['snr']:.4f} dB |\t Max Latency: {user_config['max_transmission_latency']*1000:.4f} ms |\t Acceptance: {user_config['acceptance_of_distortion']}")

        user_added_successfully = user_add_process(system_config, user_config)

        current_num = system_config.get_current_user_num()
        used_bw = system_config.get_used_bandwidth()
        total_bw = system_config.get_total_bandwidth()

        logger.info(f"System Status: Users: {current_num} | Bandwidth: {used_bw:.4f}/{total_bw:.4f} MHz.")

        if not user_added_successfully:
            logger.error(f"Failed to add user #{idx+1}. Stopping further user additions.")
            break
        
        # Periodically save user records to a CSV for analysis

        os.makedirs("Tables/User_Records", exist_ok=True)
        export_table_to_csv(
            table_name="user_record",
            csv_path=f"Tables/User_Records/user_records_numbers_{system_config.get_current_user_num()}.csv",
            db_file=config["IDA"]["database_file"]
        )
    
    logger.info("=" * 80)
    logger.info("All users from the queue have been processed.")
