import pprint
import dobble_utils as db

def main():
    X, y = db.read_and_process_image(
        'new_dataset/exp0-augmented3', nrows = 320, ncols = 240, labels = True
    )

    pprint.pprint(X)
    pprint.pprint(y)

if __name__ == '__main__':
    main()