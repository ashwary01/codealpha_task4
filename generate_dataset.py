import numpy as np
import pandas as pd

def generate_advertising_dataset():
    np.random.seed(42)
    n_samples = 200

    # Realistic Advertising spend (in $1000s)
    tv = np.random.uniform(5.0, 296.4, n_samples)
    radio = np.random.uniform(0.0, 49.6, n_samples)
    newspaper = np.random.uniform(0.3, 114.0, n_samples)

    # Categorical features matching prompt requirements
    segments = ['Mass Market', 'Gen Z & Youth', 'B2B & Professionals', 'High Income']
    platforms = ['Omnichannel Digital', 'Streaming & TV', 'Social Media', 'Print & Radio']

    target_segment = np.random.choice(segments, size=n_samples, p=[0.4, 0.3, 0.15, 0.15])
    platform = np.random.choice(platforms, size=n_samples, p=[0.35, 0.35, 0.20, 0.10])

    # Segment multipliers
    segment_weights = {
        'Mass Market': 0.0,
        'Gen Z & Youth': 0.8,
        'B2B & Professionals': 1.5,
        'High Income': 2.2
    }

    platform_weights = {
        'Omnichannel Digital': 1.2,
        'Streaming & TV': 1.0,
        'Social Media': 0.6,
        'Print & Radio': -0.4
    }

    seg_effect = np.array([segment_weights[s] for s in target_segment])
    plat_effect = np.array([platform_weights[p] for p in platform])

    # Ground truth formula with small interaction (TV & Radio synergy) and Gaussian noise
    base_sales = 2.9
    tv_coef = 0.0457
    radio_coef = 0.1885
    news_coef = 0.0010
    synergy_coef = 0.0011 # Interaction between TV & Radio spend

    noise = np.random.normal(0, 1.2, n_samples)

    sales = (
        base_sales 
        + tv_coef * tv 
        + radio_coef * radio 
        + news_coef * newspaper 
        + synergy_coef * (tv * radio / 10.0)
        + seg_effect 
        + plat_effect 
        + noise
    )

    # Round appropriately
    df = pd.DataFrame({
        'TV': np.round(tv, 1),
        'Radio': np.round(radio, 1),
        'Newspaper': np.round(newspaper, 1),
        'Target_Segment': target_segment,
        'Platform': platform,
        'Sales': np.round(np.clip(sales, 3.0, 35.0), 2)
    })

    df.to_csv('Advertising.csv', index=False)
    print(f"Successfully generated Advertising.csv with {len(df)} rows and {len(df.columns)} columns.")
    print(df.head())

if __name__ == '__main__':
    generate_advertising_dataset()
