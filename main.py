from graph.build_graph import build_graph
from state.trading_state import TradingState

if __name__ == "__main__":
    # 1. Initialize the graph
    app = build_graph()

    # 2. Graph Visualization
    try:
        # Requires: pip install pygraphviz or mermaid-python
        app.get_graph().draw_mermaid_png(output_file_path="graph.png")
        print("✅ Graph visualization saved to graph.png")
    except Exception as e:
        print(f"ℹ️ Note: Could not generate visualization: {e}")
        print("Continuing with execution...")

    # 3. Invoke the graph (Strictly typed as TradingState)
    # Using Jan 1, 2018 as our test entry point
    initial_state: TradingState = {
        "target_date": "01/01/2018",
        "currency_pair": "USDJPY",
        "ce_output": {},
        "tts_output": {},
        "siv_output": {},
        "verdict": "",
        "debug_log": ["Main: Starting single-run execution"]
    }

    print(f"--- Running Analysis for {initial_state['currency_pair']} on {initial_state['target_date']} ---")
    
    result = app.invoke(initial_state)

    # 4. Print Results
    print("\n" + "="*30)
    print("FINAL VERDICT:")
    print("="*30)
    print(result.get("verdict", "No verdict generated."))

    print("\nDEBUG TRACE:")
    print("-" * 30)
    for log in result.get("debug_log", []):
        print(log)
    print("-" * 30)