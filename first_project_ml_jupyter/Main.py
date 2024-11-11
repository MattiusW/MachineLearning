from LoadData import load_test_set


def main():
    train_set, test_set = load_test_set()

    print(len(train_set))
    print(len(test_set))

if __name__ == "__main__":
    main()