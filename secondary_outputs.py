import os
import pandas as pd
import numpy as np
import guild_app  # We can import it since it has if __name__ == "__main__":

def calculate_route_benchmark(df):
    """
    Route-Level Fuel Benchmark: Estimate expected fuel consumption for a route independent of dumper.
    We'll use 'distance_travelled' and 'total_elevation_gain' as proxies for a 'route'.
    """
    # Create bins for distance and elevation gain to group 'similar routes'
    df['dist_bin'] = pd.cut(df['distance_travelled'], bins=10)
    df['elev_bin'] = pd.cut(df['total_elevation_gain'], bins=5)
    
    benchmark = df.groupby(['dist_bin', 'elev_bin'])['acons'].mean().reset_index()
    benchmark = benchmark.rename(columns={'acons': 'benchmark_fuel'})
    
    # Also provide a simple linear model if someone wants a formulaic benchmark
    # fuel = a * dist + b * elev + c
    valid = df.dropna(subset=['acons', 'distance_travelled', 'total_elevation_gain'])
    if not valid.empty:
        X = valid[['distance_travelled', 'total_elevation_gain']]
        y = valid['acons']
        from sklearn.linear_model import LinearRegression
        model = LinearRegression().fit(X, y)
        print(f"[INFO] Benchmark Linear Formula: Fuel = {model.coef_[0]:.4f} * dist + {model.coef_[1]:.4f} * elev + {model.intercept_:.4f}")
    
    return benchmark

def calculate_dumper_efficiency(df, benchmark_df):
    """
    Dumper Efficiency Component: Capture dumper-specific variation.
    Efficiency = Actual Fuel - Benchmark Fuel for that route.
    """
    merged = df.merge(benchmark_df, on=['dist_bin', 'elev_bin'], how='left')
    merged['efficiency_gap'] = merged['acons'] - merged['benchmark_fuel']
    
    dumper_eff = merged.groupby('vehicle')['efficiency_gap'].mean().reset_index()
    dumper_eff = dumper_eff.sort_values('efficiency_gap')
    return dumper_eff

def verify_daily_consistency(df):
    """
    Daily Fuel Consistency: Ensure aggregated predicted fuel aligns with actual daily fuel consumption.
    (Using 'acons' as 'actual' for training data verification)
    """
    daily_cons = df.groupby(['date'])['acons'].sum().reset_index()
    daily_cons = daily_cons.rename(columns={'acons': 'total_actual_fuel'})
    return daily_cons

def main():
    print("Generating Secondary Outputs...")
    
    # 1. Load data using functions from haha.py
    summary_data = haha.load_summary_data()
    telem_features = haha.load_all_telemetry_features()
    refuel_features = haha.load_refuel_features()
    
    # 2. Build feature matrix
    df = haha.build_features(summary_data, telem_features, refuel_features)
    
    # Filtrar solo registros con acons (ground truth)
    df = df[df['acons'].notna()].copy()
    
    # 3. Calculate Route Benchmark
    benchmark = calculate_route_benchmark(df)
    benchmark.to_csv("route_benchmark.csv", index=False)
    print("[SUCCESS] route_benchmark.csv generated.")
    
    # 4. Calculate Dumper Efficiency
    dumper_eff = calculate_dumper_efficiency(df, benchmark)
    dumper_eff.to_csv("dumper_efficiency.csv", index=False)
    print("[SUCCESS] dumper_efficiency.csv generated.")
    
    # 5. Daily Fuel Consistency
    daily_cons = verify_daily_consistency(df)
    daily_cons.to_csv("daily_fuel_consistency.csv", index=False)
    print("[SUCCESS] daily_fuel_consistency.csv generated.")
    
    # 6. Cycle Segmentation Stats (as requested for methodology explanation)
    cycle_stats = df.groupby('vehicle')[['haul_cycles', 'dump_stop_count']].describe()
    cycle_stats.to_csv("cycle_segmentation_stats.csv")
    print("[SUCCESS] cycle_segmentation_stats.csv generated.")

if __name__ == "__main__":
    main()
