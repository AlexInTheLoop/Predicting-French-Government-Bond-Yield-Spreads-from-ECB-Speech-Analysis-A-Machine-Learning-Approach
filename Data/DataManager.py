import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).parent

class DataManager:
    def __init__(self, speeches_csv, rates_csv, output_csv="dataset.csv",
                 start_date="2011-01-01", end_date=None, force_binary=False):
        self.speeches_csv = BASE_DIR / speeches_csv
        self.rates_csv = BASE_DIR / rates_csv
        self.output_csv = BASE_DIR / output_csv
        self.start_date = pd.Timestamp(start_date)
        self.end_date = end_date or pd.Timestamp.today().normalize()
        self.force_binary = force_binary
        self.speeches_df = pd.DataFrame()
        self.rates_df = pd.DataFrame()
        self.final_df = pd.DataFrame()

    def load_speeches(self):
        df = pd.read_csv(self.speeches_csv, sep="|")
        if "date" not in df.columns:
            for cand in ["Date", "DATE", "timestamp", "Timestamp"]:
                if cand in df.columns:
                    df.rename(columns={cand: "date"}, inplace=True)
                    break
        if "date" not in df.columns:
            raise ValueError("Impossible to find the column 'date' in the speeches CSV.")

        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"])
        df = df[(df["date"] >= self.start_date) & (df["date"] <= self.end_date)]
        self.speeches_df = df.sort_values("date")


    def load_rates(self):
        df = pd.read_csv(self.rates_csv)
        if df.columns[0] != "date":
            df.rename(columns={df.columns[0]: "date"}, inplace=True)
        required_cols = ["FRYC2Y10 Index", "FRYC1030 Index"]
        for c in required_cols:
            if c not in df.columns:
                raise ValueError(f"Column {c} is missing in {self.rates_csv}")
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"])
        self.rates_df = df.sort_values("date").set_index("date")

    def _get_prev_and_curr_dates(self, market_index, speech_date):
        prev_pos = market_index.searchsorted(speech_date, side="left") - 1
        prev_date = market_index[prev_pos] if prev_pos >= 0 else None
        curr_pos = market_index.searchsorted(speech_date, side="left")
        curr_date = market_index[curr_pos] if curr_pos < len(market_index) else None
        return prev_date, curr_date

    def _sign_label(self, delta):
        if delta > 0:
            return 1
        elif delta < 0:
            return -1
        else:
            return 0 if not self.force_binary else 0

    def build_dataset(self):
        if self.speeches_df.empty:
            raise ValueError("Speeches are not loaded.")
        if self.rates_df.empty:
            raise ValueError("Rates data are not loaded.")

        out_rows = []
        mkt_index = self.rates_df.index

        for _, row in self.speeches_df.iterrows():
            d = row["date"]
            prev_date, curr_date = self._get_prev_and_curr_dates(mkt_index, d)
            if (prev_date is None) or (curr_date is None):
                continue

            prev_2y10y = self.rates_df.loc[prev_date, "FRYC2Y10 Index"]
            curr_2y10y = self.rates_df.loc[curr_date, "FRYC2Y10 Index"]
            prev_1030 = self.rates_df.loc[prev_date, "FRYC1030 Index"]
            curr_1030 = self.rates_df.loc[curr_date, "FRYC1030 Index"]

            delta_2y10y = curr_2y10y - prev_2y10y
            delta_1030 = curr_1030 - prev_1030

            entry = {
                "date": d,
                "target_1": self._sign_label(delta_2y10y),
                "target_2": self._sign_label(delta_1030),
            }

            for col in self.speeches_df.columns:
                if col not in ["date"]:
                    entry[col] = row[col]

            out_rows.append(entry)

        self.final_df = pd.DataFrame(out_rows).set_index("date").sort_index()


    def export_dataset(self):
        if self.final_df.empty:
            raise ValueError("The final dataset is empty.")
        self.final_df.to_csv(self.output_csv)
        print(f"Final dataset exported to {self.output_csv} ({len(self.final_df)} rows).")

    def summary(self):
        if self.final_df.empty:
            print("Final dataset is empty.")
            return
        print("Summary of classes (target_1):")
        print(self.final_df["target_1"].value_counts())
        print("\nSummary of classes (target_2):")
        print(self.final_df["target_2"].value_counts())