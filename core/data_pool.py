import random

class DataPool:
    def __init__(self, data):
        self.data = data
        self.modified_data = dict()

    def get_batch(self, batch_size):
        keys = random.sample(list(self.data.keys()), min(batch_size, len(self.data)))
        batch = {key: self.data[key] for key in keys}
        return batch

    def commit(self):
        self.data.update(self.modified_data)
        self.modified_data.clear()

# Example usage:
if __name__ == "__main__":
    # Initialize DataPool with some predefined data
    initial_data = {'item1': 10, 'item2': 20, 'item3': 30, 'item4': 40}
    data_pool = DataPool(initial_data)

    # Get a batch of data
    batch_size = 2
    batch = data_pool.get_batch(batch_size)
    print("Batch:", batch)

    # Modify the batch and commit the changes
    for key, value in batch.items():
        data_pool.modified_data[key] = value * 2

    data_pool.commit()

    # Print the updated data
    print("Updated Data:", data_pool.data)