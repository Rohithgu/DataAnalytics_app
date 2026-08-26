from analyzer import DataAnalyzer


class DataChatbot:

    def __init__(self, file_path):
        self.analyzer = DataAnalyzer(
            file_path
        )

    def ask(self, question):

        question = question.lower()

        if "summary" in question:
            return self.analyzer.summary()

        if "statistics" in question:
            return self.analyzer.statistics()

        if "columns" in question:
            return self.analyzer.columns()

        if "search" in question:
            words = question.split()

            if len(words) > 1:
                keyword = words[-1]

                return self.analyzer.search(
                    keyword
                )

        return {
            "message": (
                "I can help with summary, "
                "statistics, columns and search."
            )
        }