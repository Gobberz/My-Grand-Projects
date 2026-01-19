# utils/calc_helpers.py
import numpy as np
from scipy.stats import norm

def z_test_conversion(n1, c1, n2, c2):
    p1 = c1 / n1
    p2 = c2 / n2
    p_pool = (c1 + c2) / (n1 + n2)
    se = np.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
    z = (p2 - p1) / se
    p_val = 2 * (1 - norm.cdf(abs(z)))
    return {
        "p1": p1,
        "p2": p2,
        "diff": p2 - p1,
        "z": z,
        "p_value": p_val
    }

def pairwise_z_test(df):
    results = []
    for i in range(len(df)):
        for j in range(i + 1, len(df)):
            g1 = df.iloc[i]
            g2 = df.iloc[j]
            result = z_test_conversion(g1['Пользователи'], g1['Конверсии'],
                                       g2['Пользователи'], g2['Конверсии'])
            results.append({
                "Группа 1": g1["Группа"],
                "Группа 2": g2["Группа"],
                "Разница": result["diff"],
                "Z-значение": result["z"],
                "p-value": result["p_value"]
            })
    return results
