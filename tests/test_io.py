import pandas as pd

from laneige_engine import io as lio


class Upload:
    def __init__(self, name, text):
        self.name, self._b = name, text.encode()

    def getvalue(self):
        return self._b


def test_tracker_style_headers_with_title_row():
    csv = ("S&OP export,,,,,,,\n"
           "Customer,SKU,Product Name,Week Commencing,Total Forecast,Total Orders,Variance,Accuracy %\n"
           "ASDA,6015745,Scotch Egg,06-07-2026,514,964,450,0.12\n"
           'JS,6031350,Pork Cocktails,13-07-2026,768,"1,514",746,0.03\n'
           "JS,6031350,Pork Cocktails,20-07-2026,768,1500,746,0.03\n")
    df, rep = lio.load(Upload("t.csv", csv), "sell_in")
    assert rep.mapping["qty"] == "Total Orders"          # forecast column is never used as actuals
    assert rep.mapping["date"] == "Week Commencing"
    assert rep.mapping["channel"] == "Customer"
    assert df["date"].iloc[0] == pd.Timestamp("2026-07-06")  # dd-mm-yyyy
    assert df["qty"].iloc[1] == 1514
    assert rep.granularity == "weekly"


def test_wide_month_layout_is_reshaped():
    wide = pd.DataFrame({"Material": ["A1", "B2"], "Description": ["Lip mask", "Cream"],
                         "Jan-25": [10, 5], "Feb-25": [12, 6], "Mar-25": [9, 7], "Total": [31, 18]})
    df, rep = lio.standardize(wide, "sell_out")
    assert rep.reshaped_from_wide
    assert len(df) == 6
    assert set(df["date"].dt.month) == {1, 2, 3}
    assert rep.granularity == "monthly"


def test_wms_excludes_blocked_stock_and_keeps_blank_eta():
    wms = pd.DataFrame({"Item Code": ["A1", "B2", "A1"], "Warehouse": ["DC1", "DC1", "DC2"],
                        "Unrestricted Stock": [100, 50, 20], "GIT": [0, 200, 0],
                        "ETA": ["", "15/10/2026", ""], "Stock Status": ["Good", "Good", "Blocked"]})
    df, rep = lio.standardize(wms, "wms")
    assert rep.mapping["eta"] == "ETA"
    assert df.loc[df["location"] == "DC2", "on_hand"].iloc[0] == 0
    assert df.loc[df["sku"] == "B2", "eta"].iloc[0] == pd.Timestamp("2026-10-15")


def test_special_date_formats():
    assert lio.parse_dates(pd.Series(["2024-W23"])).iloc[0] == pd.Timestamp("2024-06-03")
    assert lio.parse_dates(pd.Series([45292])).iloc[0] == pd.Timestamp("2024-01-01")
    assert lio.parse_dates(pd.Series(["Jan-24", "Feb-24"])).iloc[1] == pd.Timestamp("2024-02-01")
    assert lio.parse_dates(pd.Series(["31/01/2025", "01/02/2025"])).iloc[1] == pd.Timestamp("2025-02-01")


def test_granularity_detection():
    d = pd.Series(pd.date_range("2025-01-01", periods=30, freq="D"))
    assert lio.detect_granularity(d)[0] == "D"
    assert lio.detect_granularity(pd.Series(pd.date_range("2025-01-06", periods=10, freq="7D")))[0] == "W"
    assert lio.detect_granularity(pd.Series(pd.date_range("2025-01-01", periods=10, freq="MS")))[0] == "M"
