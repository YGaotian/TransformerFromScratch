import json
import numpy as np

# A handy Json dictionary to array picker


class Picker:
    """
    Picker assumes the data file will be a json dictionary of the following format:
        {
            "0":
                {
                    "string-type key 1": any-type value,
                    "string-type key 2": any-type value
                },
            "1":
                {
                    "string-type key 1": any-type value,
                    "string-type key 2": any-type value
                },
            ...
        }
    """
    def __init__(self, file_path, shuffle=False, tell=False):
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                raw_data = json.load(file)

            self._data = np.array([raw_data[str(key)] for key in range(len(raw_data))])

            if shuffle:
                rng = np.random.default_rng()
                rng.shuffle(x=self._data)
            if tell:
                print(f"{len(self._data)} data loaded")
        except FileNotFoundError:
            raise FileNotFoundError(f"Failed: no file founded: '{file_path}'")
        except json.JSONDecodeError:
            raise json.JSONDecodeError(f"Failed: '{file_path}' is not valid json file.")
        except Exception as e:
            raise e

    def __getitem__(self, key):
        if isinstance(key, (int, slice)):
            return self._data[key]

        elif isinstance(key, tuple):
            if len(key) != 2:
                raise IndexError("Only 2D and 1D indexing are supported.")

            idx, people = key
            if not isinstance(idx, (int, slice)) or not isinstance(people, str):
                raise IndexError("Please use [int | slice, str] format for 2D indexing.")

            if isinstance(idx, int):
                return self._data[idx][people]
            elif isinstance(idx, slice):
                selected_items = [item[people] for item in self._data[idx]]
                return np.array(selected_items)

    def tolist(self):
        return self._data.tolist()

    @property
    def all(self):
        return self._data

    def __repr__(self):
        return (f"Databag object containing:\n {self._data}\n\n"
                f"Use Databag.all, Databag[id | slice], "
                f"or Databag[id | slice, \"key\"] to access the actual data.")

    def __len__(self):
        return len(self._data)