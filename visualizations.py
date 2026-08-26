import pandas as pd
import matplotlib.pyplot as plt
import os


def generate_chart(
    file_path,
    column,
    chart_type="bar"
):
    df = pd.read_csv(file_path)

    os.makedirs(
        "charts",
        exist_ok=True
    )

    output_file = f"charts/{column}_{chart_type}.png"

    if chart_type == "bar":

        df[column].value_counts().head(10).plot(
            kind="bar"
        )

        plt.title(
            f"Bar Chart - {column}"
        )

    elif chart_type == "histogram":

        df[column].plot(
            kind="hist"
        )

        plt.title(
            f"Histogram - {column}"
        )

    else:
        return {
            "error": "Unsupported chart type."
        }

    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()

    return output_file