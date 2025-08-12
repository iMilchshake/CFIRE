from lxg.datasets import dataset_callables

tasks=[
    "abalone",
    "breastw",
    "spambase",
    "heloc",
    "beans",
    "ionosphere",
    "breastcancer",
    "btsc",
    "spf",
    "wine",
    "diggle",
    "iris",
    "vehicle",
    "autouniv",
]


for task in tasks:
    print(f"loading {task}")
    try:
        random_state = 42 if task in ["beans", "ionosphere", "breastcancer"] else None
        data = dataset_callables[task](random_state=random_state)
        print("\tOK")
    except SystemExit as e:
        print(f"SystemExit: \t{e}")
    except Exception as e:
        print(f"Exception: \t{e}")

print("done!")