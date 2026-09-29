from tools.minimax_client import minimax_chat


def main():
    result = minimax_chat(
        [
            {
                "role": "user",
                "content": "只回答：OK",
            }
        ]
    )

    print("RESULT:")
    print(result)


if __name__ == "__main__":
    main()