import sys
import os

sys.path.append('../../')  # This is for finding all the modules

# --- DEPENDENCY MOCKING START ---
# The LLaMEA dependency is required by llm4ad imports but we are not using LLaMEA method.
# To avoid installing the heavy/broken 'llamea' package, we mock it here.
import sys
from unittest.mock import MagicMock
mock_llamea = MagicMock()
mock_llamea.Solution = MagicMock
mock_llamea.prepare_namespace = MagicMock(return_value=({}, None))
sys.modules['llamea'] = mock_llamea
# --- DEPENDENCY MOCKING END ---

from evaluation import CirclePackingEvaluation
# from llm4ad.tools.llm.llm_api_https import HttpsApi
# Use OpenAIAPI for better compatibility with Google's Base URL
from llm4ad.tools.llm.llm_api_openai import OpenAIAPI
# Import EoH
from llm4ad.method.eoh import EoH, EoHProfiler
# Import MCTS-AHD
from llm4ad.method.mcts_ahd import MCTS_AHD
from llm4ad.method.mcts_ahd.profiler import MAProfiler
# Import ReEvo
from llm4ad.method.reevo import ReEvo, ReEvoProfiler
# Import HillClimb (Assuming this is what was meant by 'hsEvo' or as a baseline)
from llm4ad.method.hillclimb import HillClimb, HillClimbProfiler


def main():
    # Setup LLM with Gemini-2.5-Pro as requested
    # We use OpenAIAPI class to support the specific Google Base URL
    llm = OpenAIAPI(
        base_url='https://generativelanguage.googleapis.com/v1beta/openai/', 
        api_key='AIbah', 
        model='gemini-2.5-flash',
        timeout=120
    )

    # Initialize Task
    task = CirclePackingEvaluation(timeout_seconds=1200)

    # Run EoH
    for i in range(3):
        eoh_method = EoH(
            llm=llm,
            profiler=EoHProfiler(log_dir='logs/eoh', log_style='simple'),
            evaluation=task,
            max_sample_nums=500,  # Max sample nums must be 500
            max_generations=100,  # Setting a reasonable generation limit
            pop_size=10,
            num_samplers=4, # Reduced for simple run
            num_evaluators=4, # Reduced for simple run
            debug_mode=False
        )
        eoh_method.run()

        print("\n\n")
        print("========================================")
        print("Running MCTS-AHD on Circle Packing")
        print("========================================")

        # Run MCTS-AHD
        mcts_method = MCTS_AHD(
            llm=llm,
            profiler=MAProfiler(log_dir='logs/mcts_ahd', log_style='simple'),
            evaluation=task,
            max_sample_nums=500, # Max sample nums must be 500
            pop_size=10, 
            init_size=4,
            num_samplers=4,
            num_evaluators=4
        )
        mcts_method.run()
        
        print("\n\n")
        print("========================================")
        print("Running ReEvo on Circle Packing")
        print("========================================")

        # Run ReEvo
        reevo_method = ReEvo(
            llm=llm,
            profiler=ReEvoProfiler(log_dir='logs/reevo', log_style='simple'),
            evaluation=task,
            max_sample_nums=500,
            pop_size=10, 
            num_samplers=4,
            num_evaluators=4
        )
        reevo_method.run()
    
    # print("\n\n")
    # print("========================================")
    # print("Running HillClimb (hsEvo) on Circle Packing")
    # print("========================================")
    
    # # Run HillClimb
    # hc_method = HillClimb(
    #     llm=llm,
    #     profiler=HillClimbProfiler(log_dir='logs/hillclimb', log_style='simple'),
    #     evaluation=task,
    #     max_sample_nums=500,
    #     num_samplers=4,
    #     num_evaluators=4
    # )
    # hc_method.run()


if __name__ == '__main__':
    main()
