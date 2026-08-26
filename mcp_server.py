from mcp.server.fastmcp import FastMCP

from tools.analysis_tools import (
    get_dataset_summary,
    get_statistics,
    filter_data
)

from tools.keyword_tools import (
    search_keyword,
    search_column_keyword
)

from tools.sql_tools import (
    execute_sql
)

from tools.visualization_tools import (
    create_bar_chart,
    create_histogram
)


mcp = FastMCP(
    "Data Analysis MCP Server"
)


@mcp.tool()
def dataset_summary(
    file_path: str
) -> dict:
    """
    Get basic information about a CSV dataset.
    """

    return get_dataset_summary(
        file_path
    )


@mcp.tool()
def dataset_statistics(
    file_path: str
) -> dict:
    """
    Get descriptive statistics
    for numerical columns.
    """

    return get_statistics(
        file_path
    )


@mcp.tool()
def filter_dataset(
    file_path: str,
    column: str,
    value: str
) -> list:
    """
    Filter a dataset using a column
    and search value.
    """

    return filter_data(
        file_path,
        column,
        value
    )


@mcp.tool()
def keyword_search(
    file_path: str,
    keyword: str
) -> dict:
    """
    Search for a keyword across
    the entire dataset.
    """

    return search_keyword(
        file_path,
        keyword
    )


@mcp.tool()
def column_keyword_search(
    file_path: str,
    column: str,
    keyword: str
) -> dict:
    """
    Search for a keyword inside
    a specific column.
    """

    return search_column_keyword(
        file_path,
        column,
        keyword
    )


@mcp.tool()
def run_sql(
    file_path: str,
    query: str
) -> list:
    """
    Execute a SQL query against
    the CSV dataset.
    """

    return execute_sql(
        file_path,
        query
    )


@mcp.tool()
def bar_chart(
    file_path: str,
    column: str
) -> dict:
    """
    Create a bar chart for a column.
    """

    return create_bar_chart(
        file_path,
        column
    )


@mcp.tool()
def histogram(
    file_path: str,
    column: str
) -> dict:
    """
    Create a histogram for
    a numerical column.
    """

    return create_histogram(
        file_path,
        column
    )


if __name__ == "__main__":

    mcp.run(
        transport="stdio"
    )