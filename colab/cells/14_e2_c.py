display(pd.read_csv(f"{E2}/E2_effects_table.csv").query("Kind == 'primary'")[["Outcome", "Effect", "Odds ratio", "95% CI", "p-value (raw)", "p-value (Holm, primary only)"]])
print(open(f"{E2}/E2_GEE_errors.txt").read())
