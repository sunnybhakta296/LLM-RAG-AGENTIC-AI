# main.py
from graph import graph

if __name__ == "__main__":
    task = input("Enter a task: ")
    output = graph.invoke({"task": task})
    print("\n🧠 Result:\n", output["result"])


    # # Example tasks for testing
    # example_tasks = [
    #     "Summarize the latest news about AI.",
    #     "Generate a Python script to sort a list of numbers.",
    #     "Explain the concept of transfer learning.",
    #     "List three benefits of using cloud computing.",
    #     "Write a short story about a robot learning emotions.",
    #     "Generate a Python script to add 2 numbers.",
    # ]

    # print("\nExample tasks you can try:")
    # for idx, t in enumerate(example_tasks, 1):
    #     print(f"{idx}. {t}")