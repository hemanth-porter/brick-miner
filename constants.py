complaints_modelling_prompt = """
Identify the complaints or suggestions in the given review.
Each review may have multiple complaints or suggestions also. 
Check if the complaints or suggestions are similar to a complaint or suggestions which is part of this list:{}. 
If yes, return same complaint or suggestions. If not, give the complaints that you have identified. Keep it short and specific. Your response should be in this format. output format : [your list of complaints or suggestions here]. Here is an example output for your reference : ['Unexpected increase in cost', 'Inconsistent quotation'].Please stick to given format only
"""

positive_points_modelling_prompt = """
Identify the positive points in the given review.
Each review may have multiple positive points also. 
Check if the positive points are similar to a positive point which is part of this list:{}. 
If yes, return same positive points. If not, give the positive points that you have identified. Keep it short and specific. Your response should be in this format. output format : [your list of positive points here]. Here is an example output for your reference : ['Value for money','Excellent customer care service'].Please stick to given format only
"""

change_format_prompt = """
Change the format of given pints to this format.
Output format : [your list of points here]. 
Here is an example output for your reference : ['Unexpected increase in cost', 'Inconsistent quotation'].
Please stick to given format only"
"""

theme_finder_prompt = """
                As a theme-based topic grouper, your task is to group a given list of topics into high-level themes that represent similar topics while being specific in identifying the high-level topic. Your goal is to group the topics without changing their spelling or case. Your output should be in the following format:

                your_identified_theme_name : ["topic1", "topic2", "topic3"]

                Example:
                Input:
                ["good service", "very good experience", "well packed", "prompt delivery", "awesome experience","more upi payment options", "show previous completed orders","Good packing"]

                Example Output:
                {"Customer satisfaction": ["good service", "very good experience", "prompt delivery", "awesome experience"],
                 "Packaging": ["well packed","Good packing"],
                 "App suggestions":  : ["more upi payment options", "show previous completed orders"]
                }

                Please group the following topics into appropriate high-level themes without altering the spelling or the case of the individual topics.

                Note: Ensure that you include all the individual topics provided.
"""