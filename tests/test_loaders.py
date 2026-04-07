import pandas as pd

from src.data.loaders import DataLoader

def test_loader_has_known_datasets() -> None:
loader = DataLoader()



assert "base" in loader.dataset_paths
assert "variant_1" in loader.dataset_paths
assert "variant_2" in loader.dataset_paths

def test_save_dataset_to_csv(tmp_path) -> None:
loader = DataLoader()
df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})



output_path = tmp_path / "test.csv"
loader.save_dataset(df, output_path)

loaded_df = pd.read_csv(output_path)

assert loaded_df.shape == (2, 2)
assert list(loaded_df.columns) == ["a", "b"]

