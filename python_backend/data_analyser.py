import os
import pandas as pd
# pyrefly: ignore [missing-import]
import numpy as np

DATA_DIR = r"C:\Users\PRAVEEN PRABAKARN\Desktop\new fold\AGRI_AI_BACKEND\agri_data_extracted\agri_data"

class AgriDataAnalyser:
    def __init__(self):
        self.crop_data = None
        self.rainfall_data = None
        self.land_use_data = None
        self.pdf_proposal_text = ""
        self.load_data()

    def load_data(self):
        # 1. Load Crop Production Data
        crop_path = os.path.join(DATA_DIR, "Tamilnadu agriculture yield data.csv")
        if os.path.exists(crop_path):
            try:
                self.crop_data = pd.read_csv(crop_path)
                # Calculate Yield: Production / Area
                self.crop_data['Yield'] = self.crop_data['Production'] / self.crop_data['Area']
            except Exception as e:
                print(f"Error loading crop data: {e}")
        else:
            print("Crop data file not found.")

        # 2. Load Rainfall Data
        rainfall_path = os.path.join(DATA_DIR, "rainfall_data.csv")
        if os.path.exists(rainfall_path):
            try:
                self.rainfall_data = pd.read_csv(rainfall_path, encoding='utf-8-sig')
            except Exception as e:
                print(f"Error loading rainfall data: {e}")

        # 3. Load Land Use Data
        land_use_path = os.path.join(DATA_DIR, "land_use.csv")
        if os.path.exists(land_use_path):
            try:
                self.land_use_data = pd.read_csv(land_use_path, encoding='utf-8-sig')
            except Exception as e:
                print(f"Error loading land use data: {e}")

        # 4. Load PDF proposal text if extracted
        proposal_txt_path = r"C:\Users\PRAVEEN PRABAKARN\Desktop\new fold\AGRI_AI_BACKEND\agri_pdf_text.txt"
        if os.path.exists(proposal_txt_path):
            try:
                with open(proposal_txt_path, "r", encoding="utf-8") as f:
                    self.pdf_proposal_text = f.read()
            except Exception as e:
                print(f"Error reading proposal text: {e}")
        else:
            # Fallback inline content of proposal
            self.pdf_proposal_text = """
AGRISMART – SMART FARMING SYSTEM WITH AI  
Overview:  
This project aims to revolutionize traditional agriculture by introducing a Smart Farming System 
(AgriSmart) that leverages Artificial Intelligence (AI) and Internet of Things (IoT) technologies. The 
system will provide farmers with real-time data on soil health, weather conditions, and crop 
growth to help them make informed decisions and optimize their crop yields.  
Features:  
1. Soil Analysis and Fertilizer Recommendation: The system analyzes soil parameters like N, P, K, pH, moisture, and temperature to recommend suitable crops and fertilizers.
2. Crop Disease Detection: Uses an image analysis model to analyze crop images, identify diseases, and provide treatment solutions.
3. Market Trends Analysis: Real-time response from the internet regarding current crop prices, demand, and market trends.
4. AI Assistant: Conversational assistant providing intelligence to farmers.
"""

    def get_district_crop_stats(self, district: str, crop: str = None):
        """Get production, area, and yield stats for a district and optionally a specific crop."""
        if self.crop_data is None:
            return {"error": "Crop data not loaded"}

        dist_df = self.crop_data[self.crop_data['District_Name'].str.upper() == district.upper()]
        if dist_df.empty:
            return {"error": f"No data found for district '{district}'"}

        if crop:
            crop_df = dist_df[dist_df['Crop'].str.upper() == crop.upper()]
            if crop_df.empty:
                return {
                    "district": district,
                    "crop": crop,
                    "message": f"No data found for crop '{crop}' in district '{district}'",
                    "available_crops": list(dist_df['Crop'].unique())
                }
            
            stats = {
                "district": district,
                "crop": crop,
                "avg_area": float(crop_df['Area'].mean()),
                "avg_production": float(crop_df['Production'].mean(skipna=True)),
                "avg_yield": float(crop_df['Yield'].mean(skipna=True)),
                "years_covered": [int(y) for y in crop_df['Crop_Year'].unique()],
                "seasons": list(crop_df['Season'].unique())
            }
            return stats
        else:
            # Aggregate stats across all crops in district
            top_crops = dist_df.groupby('Crop')['Production'].mean().sort_values(ascending=False).head(5)
            stats = {
                "district": district,
                "total_records": len(dist_df),
                "top_crops_by_production": {k: float(v) for k, v in top_crops.items() if not np.isnan(v)},
                "unique_crops_count": int(dist_df['Crop'].nunique()),
                "seasons": list(dist_df['Season'].unique())
            }
            return stats

    def get_crop_recommendations_by_district(self, district: str):
        """Recommend crops for a district based on historical production and yield."""
        if self.crop_data is None:
            return []

        dist_df = self.crop_data[self.crop_data['District_Name'].str.upper() == district.upper()]
        if dist_df.empty:
            return []

        # Group by Crop and get average yield and production
        grouped = dist_df.groupby('Crop').agg(
            avg_yield=('Yield', 'mean'),
            avg_production=('Production', 'mean'),
            avg_area=('Area', 'mean')
        ).reset_index()

        # Sort by yield and production
        grouped = grouped.sort_values(by=['avg_yield', 'avg_production'], ascending=[False, False])
        
        recommendations = []
        for _, row in grouped.head(8).iterrows():
            recommendations.append({
                "crop": row['Crop'],
                "avg_yield": float(row['avg_yield']) if not pd.isna(row['avg_yield']) else 0.0,
                "avg_production": float(row['avg_production']) if not pd.isna(row['avg_production']) else 0.0,
                "avg_area": float(row['avg_area']) if not pd.isna(row['avg_area']) else 0.0
            })
        return recommendations

    def get_rainfall_summary(self, district: str):
        """Retrieve rainfall history/summary for a district if available."""
        if self.rainfall_data is None:
            return {"message": "Rainfall data not available"}
        
        # Simple lookup in rainfall columns
        # The columns are: Sl.No., Unnamed: 1, SOUTH - WEST MONSOON, etc.
        # Let's search if the district is in the dataframe
        # Let's inspect rows
        dist_rows = self.rainfall_data[self.rainfall_data.iloc[:, 1].astype(str).str.upper() == district.upper()]
        if dist_rows.empty:
            # Let's search if it is in the first column or anywhere
            for col in self.rainfall_data.columns:
                match = self.rainfall_data[self.rainfall_data[col].astype(str).str.upper() == district.upper()]
                if not match.empty:
                    dist_rows = match
                    break
        
        if dist_rows.empty:
            return {"message": f"No rainfall record found for {district}"}
        
        # Format the row details
        row_dict = dist_rows.iloc[0].to_dict()
        clean_row = {str(k): (float(v) if isinstance(v, (int, float)) and not pd.isna(v) else str(v)) for k, v in row_dict.items() if not pd.isna(v)}
        return clean_row

    def compare_yield_efficiency(self, district: str, crop: str, farmer_area: float, farmer_production: float):
        """Compare a farmer's yield to local historical averages."""
        farmer_yield = farmer_production / farmer_area if farmer_area > 0 else 0
        stats = self.get_district_crop_stats(district, crop)
        
        if "error" in stats or "message" in stats:
            return {
                "farmer_yield": farmer_yield,
                "average_yield": None,
                "efficiency_ratio": None,
                "comparison": "No local historical data available for comparison."
            }

        avg_yield = stats["avg_yield"]
        if pd.isna(avg_yield) or avg_yield == 0:
            return {
                "farmer_yield": farmer_yield,
                "average_yield": None,
                "efficiency_ratio": None,
                "comparison": "Yield comparison not available due to missing historical yield data."
            }

        efficiency = farmer_yield / avg_yield
        percent_diff = (efficiency - 1) * 100
        
        if percent_diff > 10:
            comparison = f"Your yield ({farmer_yield:.2f} t/ha) is {percent_diff:.1f}% HIGHER than the district average of {avg_yield:.2f} t/ha. Great work!"
        elif percent_diff < -10:
            comparison = f"Your yield ({farmer_yield:.2f} t/ha) is {abs(percent_diff):.1f}% LOWER than the district average of {avg_yield:.2f} t/ha. Consider checking soil parameters and applying optimized fertilizers."
        else:
            comparison = f"Your yield ({farmer_yield:.2f} t/ha) is close to the district average of {avg_yield:.2f} t/ha."

        return {
            "farmer_yield": farmer_yield,
            "average_yield": avg_yield,
            "efficiency_ratio": efficiency,
            "percent_difference": percent_diff,
            "comparison": comparison,
            "district_crop_stats": {
                "avg_area": stats["avg_area"],
                "avg_production": stats["avg_production"]
            }
        }
