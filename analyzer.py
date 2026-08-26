import pandas as pd

from tools.analysis_tools import (
    get_dataset_summary,
    get_statistics,
    filter_data
)

from tools.keyword_tools import (
    search_keyword
)


class DataAnalyzer:

    def __init__(self, file_path):
        self.file_path = file_path
        self.data = pd.read_csv(file_path)

    def summary(self):
        return get_dataset_summary(
            self.file_path
        )

    def statistics(self):
        return get_statistics(
            self.file_path
        )

    def search(self, keyword):
        return search_keyword(
            self.file_path,
            keyword
        )

    def filter(self, column, value):
        return filter_data(
            self.file_path,
            column,
            value
        )

    def columns(self):
        return list(self.data.columns)