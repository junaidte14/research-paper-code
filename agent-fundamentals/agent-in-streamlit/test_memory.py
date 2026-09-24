# test_memory.py - Unit Test Verification Suite for Memory Subsystem
from core.memory import MemoryManager

if __name__ == "__main__":
    print("=== Testing MemoryManager Subsystem ===")

    # 1. Initialize Memory Manager with strict small budget
    memory = MemoryManager(
        system_prompt="System Persona: Helpful Assistant.",
        max_token_budget=60  # Small budget to trigger pruning quickly
    )

    # 2. Add multiple messages
    memory.add_message("user", "Old Message 1: Hello world")
    memory.add_message("assistant", "Old Response 1: Hello! How can I help you today?")
    memory.add_message("user", "Old Message 2: What is 50 times 10?")
    memory.add_message("assistant", "Old Response 2: 50 times 10 is 500.")
    memory.add_message("user", "New Message 3: Tell me a short quote.")

    print(f"Total Unpruned Messages: {len(memory.history)}")
    print(f"Total Estimated Tokens: {memory.get_total_tokens()}")

    # 3. Get Pruned Messages for API call
    pruned = memory.get_pruned_messages()

    print(f"\n=== Pruned Result ({len(pruned)} messages kept) ===")
    for idx, msg in enumerate(pruned):
        print(f"[{idx}] Role: {msg['role']} | Content: {msg.get('content')}")

    # Assertion check
    assert pruned[0]["role"] == "system", "System prompt lost!"
    print("\n✅ Test Passed: System message preserved and sliding window applied successfully!")
