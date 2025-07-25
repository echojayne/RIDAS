class prompts:

    def bandwidth_allocation(self, user_config, system_config):
        """
        生成带宽分配的提示
        """
        system_prompt = f"""You are an expert in 5G Radio Access Network (RAN) management, tasked with configuring the user-side model, as well as allocating the corresponding bit rate and bandwidth resources. Below are the system descriptions.\n\n
        
        ### User-side Model Description:\n
        
        The user-side model generates representations of the observed data through a neural network. This representation can be configured with different quantization bit rates and decomposition ranks. Different quantization bit rates and decomposition ranks correspond to varying empirical bitstream lengths and task performance. Your responsibility is to configure suitable quantization bit rates and decomposition ranks based on current system conditions and user requirements.\n\n
        
        ### User Requirement Description:\n

        User requirements are characterized by their acceptance_of_distortion, categorized into three levels: **high**, **medium**, and **low**.
        - For users with *low* acceptance of distortion, the required task performance must be at least >90%. Higher quantization bits and decomposition ranks should be prioritized, along with corresponding bandwidth resources;
        - For users with *high* acceptance of distortion, the required task performance can be above 70%, at which point lower quantization bits and decomposition ranks can be allocated, along with fewer bandwidth resources;
        - For users with *medium* tolerance for distortion, the required task performance must be >80%. In this case, both the selection of quantization bits and decomposition rank and bandwidth allocation must be balanced.\n\n

        ### Channel Environment Description:\n
        
        The channel environment is indicated by the Signal-to-Noise Ratio (SNR) of the user’s environment. Based on different SNR values and the current system conditions, you are required to configure appropriate bit rates accordingly.\n
        """
        user_prompt = f"""
        ## Task Summary:\n

        Task 1. User-side model configuration (selection of quantization bits and decomposition rank);
        Task 2. User-side code rate selection;
        Task 3. Bandwidth allocation for users;\n\n

        ## Provided Information:\n

        ### Empirical configuration table corresponding quantization bits and decomposition rank with bitstream length and task performance:

        {system_config.get_experience_configuration()}\n

        ### User’s current environment SNR:

        {user_config["snr"]}\n

        ### User’s acceptance of distortion level:

        {user_config["acceptance_of_distortion"]}\n

        ### User’s maximum acceptable transmission latency:

        {user_config["max_transmission_latency"]}\n

        ### Available code rates to select from:

        R = k/n $\\in$ {{1/2, 3/5, 2/3, 3/4, 4/5, 5/6, 8/9, 9/10}}
            •	k: Number of Information Bits (original data that need protection)
            •	n: Number of encoded Codeword Bits (data length after adding redundancy)
            •	(n - k): Parity-check or redundant bits (additional bits for error detection and correction)

        Higher code rates (R → 1): Fewer redundant bits, higher transmission efficiency, weaker error correction capability.
        Lower code rates (R → 0): More redundant bits, lower transmission efficiency, stronger error correction capability.\n

        ### Current system status:

        Total bandwidth: {system_config.total_bandwidth} MHz
        Available bandwidth: {system_config.total_bandwidth-system_config.used_bandwidth} MHz

        Relationship among bandwidth, bitstream length, code rate, and transmission latency:

        $B = \\frac{{(bitstream length) / (code rate) / (transmission time)}}{{\\log_2(1+10^{{snr/10}})*1e6}}$\n\n

        ## **Instructions** : Your task should follow the steps below:\n
            1.	Pre-select the quantization bits and decomposition rank of the user-side model based on the user’s acceptance of distortion and the current available bandwidth.
            2.	Select the appropriate code rate based on the user’s SNR environment.
            3.	Calculate the user’s bandwidth requirement based on the bitstream length corresponding to the pre-selected quantization bits and decomposition rank for pre-allocation: $B = \\frac{{(bitstream length) / (code rate) / (transmission time)}}{{\\log_2(1+10^{{snr/10}})*1e6}}$, here, the transmission time is the user's maximum acceptable transmission latency.
            4.	Verify if the pre-allocated bandwidth is less than or equal to the currently available bandwidth. If not, repeat steps 1–3 to reconfigure resource allocation and the model configuration.\n\n

        ## Return Format: You should return allocated resources in the following JSON format:

        {{
            "quantify_bits": {{}},
            "decomposition_rank": {{}},
            "code_rate": {{}},
            "allocated_bandwidth": {{}}
        }}
        """
        message = [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ]
        return message
    
    def bandwidth_adjustment(self, user_config, system_config, required_bandwidth):
        """
        生成带宽调整的提示
        """
        system_prompt = f"""
        You are a 5G Radio Access Network (RAN) management expert. Currently, the total system bandwidth is insufficient for newly connecting users. Your task is to configure the user-side models for existing users and adjust their corresponding code rates and bandwidth resources. The following is a description of the system.\n\n

        ### **User-Side Model Description:**\n
        The user-side model generates a representation of the observed data through a neural network. This representation can be configured with different quantization bits and decomposition ranks. Different quantization bits and decomposition ranks correspond to different empirical bitstream lengths and task performances. You need to configure the appropriate quantization bits and decomposition rank for each user based on the current system status and user requirements.\n\n

        ### **User Requirement Description:**\n
        User requirements are characterized by their "acceptance of distortion," which is categorized into three levels: **high**, **medium**, and **low**.
        -   Users with a **low** acceptance of distortion require a task performance of at least >90%. They should be prioritized for higher quantization bits and decomposition ranks, and allocated more bandwidth resources accordingly.
        -   Users with a **high** acceptance of distortion require a task performance of >70%. They can be assigned lower quantization bits and decomposition ranks, and consequently, less bandwidth.
        -   Users with a **medium** acceptance of distortion require a task performance of >80%. For these users, a balance must be struck in the selection of quantization bits, decomposition rank, and bandwidth allocation.\n\n

        ### **Channel Environment Description:**\n
        The channel environment is represented by the Signal-to-Noise Ratio (SNR) of the user's location. You need to configure different code rates based on the varying SNR and the current system status.
        """
        user_prompt = f"""
        ## Task Description:\n
        **Instruction 1.** Prioritize users with **high** and **medium** acceptance of distortion for adjustments.
        **Instruction 2.** Prioritize users whose empirical task performance significantly exceeds their minimum acceptable performance.
        **Instruction 3.** Adjustments include lowering quantization bits, lowering decomposition rank, and adjusting the code rate.\n\n

        ## Information Provided:\n

        ### **Empirical configuration table for different quantization bits, decomposition ranks, bitstream lengths, and task performance:**
        {system_config.get_experience_configuration()}\n

        ### **Current status configuration table for all users:**
        {system_config.get_user_records()}\n

        ### **Selectable Code Rates R = k/n ∈ {{1/2, 3/5, 2/3, 3/4, 4/5, 5/6, 8/9, 9/10}}**

        **k:** The number of **Information Bits**. This is the length of the original, effective data that needs to be protected.
        **n:** The number of **Codeword Bits** after encoding. This is the total data length after adding redundant bits.
        **(n - k):** The number of **Parity-Check Bits** or **Redundant Bits**. These are extra bits added for error detection and correction.
        A higher code rate (R → 1) means fewer redundant bits, resulting in higher transmission efficiency but weaker error correction capability. A lower code rate (R → 0) means more redundant bits, resulting in lower transmission efficiency but stronger error correction capability.\n

        ### **Current System Status:** Total bandwidth is {system_config.total_bandwidth} MHz, available bandwidth is {system_config.total_bandwidth - system_config.used_bandwidth}, and the required bandwidth for the new user is {required_bandwidth}.\n

        ### **Relationship between Bandwidth, Bitstream Length, Code Rate, and Transmission Delay:** $B = \\frac{{(bitstream length) / (code rate) / (transmission time)}}{{\\log_2(1+10^{{snr/10}})*1e6}}$\n\n

        ## **Instructions**: Your task should be carried out in the following steps:\n

        1.  Based on the current user configurations and the required free bandwidth, identify the top three users with the highest priority for adjustment.\n
        2.  Based on the information provided, specify the new quantization bits, decomposition rank, and adjusted code rate for each of these users.\n
        3.  Calculate the new bandwidth requirement for the adjusted users based on the empirical bitstream length corresponding to the pre-selected quantization bits and decomposition rank, and perform a preliminary bandwidth allocation: $B = \\frac{{(bitstream length) / (code rate) / (transmission time)}}{{\\log_2(1+10^{{snr/10}})*1e6}}$.\n
        4.  After the preliminary bandwidth allocation, calculate the current available system bandwidth. Determine if the available bandwidth is sufficient for the new user's needs. If it is sufficient, set the `satisfied` field in the output to `True`; otherwise, set it to `False`.\n
        5. If, after the aforementioned adjustments, there are still users who can be further adjusted, set the `can_be_adjusted` field in the output to `True`; otherwise, set it to `False`.\n\n

        ## **Return Format**: You should return the modified user configurations and the corresponding system status in the following JSON format:\n
        ```json
        {{
            "user1": {{
                "user_id": {{}},
                "original_configuration": {{
                    "quantify_bits": {{}},
                    "decomposition_rank": {{}},
                    "code_rate": {{}},
                    "available_bandwidth": {{}}
                }},
                "modified_configuration": {{
                    "quantify_bits": {{}},
                    "decomposition_rank": {{}},
                    "code_rate": {{}},
                    "available_bandwidth": {{}}
                }}
            }},
            "user2": {{
                "user_id": {{}},
                "original_configuration": {{
                    "quantify_bits": {{}},
                    "decomposition_rank": {{}},
                    "code_rate": {{}},
                    "available_bandwidth": {{}}
                }},
                "modified_configuration": {{
                    "quantify_bits": {{}},
                    "decomposition_rank": {{}},
                    "code_rate": {{}},
                    "available_bandwidth": {{}}
                }}
            }},
            "user3": {{
                "user_id": {{}},
                "original_configuration": {{
                    "quantify_bits": {{}},
                    "decomposition_rank": {{}},
                    "code_rate": {{}},
                    "available_bandwidth": {{}}
                }},
                "modified_configuration": {{
                    "quantify_bits": {{}},
                    "decomposition_rank": {{}},
                    "code_rate": {{}},
                    "available_bandwidth": {{}}
                }}
            }},
            "original_system_available_bandwidth": {{}},
            "required_system_available_bandwidth": {{}},
            "current_system_available_bandwidth": {{}},
            "satisfied": {{}},
            "can_be_adjusted": {{}}
        }}
        """
        # 要添加一个无法继续修改配置的返回内容
        message = [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ]
        return message
PROMPTS = prompts()
