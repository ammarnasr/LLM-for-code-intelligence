import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stop_token', type=str, default='EOS')

    args = parser.parse_args()

    stop_tokens = args.stop_token.split(',')

    print(stop_tokens)