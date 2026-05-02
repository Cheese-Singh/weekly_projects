import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import os

def exploratory_data_analysis(csv_file, output_pdf="EDA_Report.pdf"):
    if not os.path.exists(csv_file):
        print(f"Error: {csv_file} not found!")
        return
    
    df = pd.read_csv(csv_file)
    df_numeric = df.select_dtypes(include='number')
    
    if df_numeric.empty:
        print("No numerical data in this dataset")
        return
    
    print(f"Generating report for {len(df_numeric.columns)} features...")
    
    try:
        with PdfPages(output_pdf) as pdf:

            for col in df_numeric.columns:
                plt.figure()
                sns.histplot(df_numeric[col], bins=30, kde=True)
                plt.title(f"Histogram of {col}")
                plt.xlabel(col)
                plt.ylabel("Frequency")
                plt.grid(axis='y', color='gray', linestyle='--', alpha=0.5)
                plt.tight_layout()
                pdf.savefig()
                plt.close()

            if df_numeric.shape[1]>1:
                corr_matrix = df_numeric.corr()
            
                plt.figure(figsize=(10, 8))
                sns.heatmap(
                    corr_matrix,
                    annot=True,
                    cmap="coolwarm",
                    fmt=".2f",
                    linewidths=0.5
                )
                plt.title("Correlation Heatmap")
                plt.tight_layout
                pdf.savefig()
                plt.close()

            print(f"Successfully saved EDA report to: {output_pdf}")

    except Exception as e:
        print(f"An error occurred during PDF generation: {e}")

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(current_dir, "eda_outputs")
    os.makedirs(output_dir, exist_ok=True)

    input_csv = os.path.join(current_dir, "Heart_Disease_Prediction.csv")
    if not os.path.exists(input_csv):
        input_csv = "/Users/ekamveersingh/Downloads/datasets/Heart_Disease_Prediction.csv"

    output_path = os.path.join(output_dir, "EDA_Report.pdf")
    
    exploratory_data_analysis(input_csv, output_path)