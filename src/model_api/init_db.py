from model_api.dataloaders import DataLoader


def init_db():
    _ = DataLoader()


if __name__ == '__main__':
    init_db()