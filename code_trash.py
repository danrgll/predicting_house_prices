# Setze manuell die Zahlen der Kategorien
set = [["MSZoning", ["A", "C(all)", "C", "FV", "I", "RH", "RL", "RP", "RM", "NI"]],
       ["LotShape", ["Reg", "IR1", "IR2", "IR3"]],
       ["LandContour", ["Low", "HLS", "Bnk", "Lvl"]],

       ]
dict_encoders = encoder.set_manual_encoder(set)
dict_encoder["MSZoning"] = create_fit_encoder(["A", "C(all)", "C", "FV", "I", "RH", "RL", "RP", "RM"])
dict_encoder["LotShape"] = create_fit_encoder(["Reg", "IR1", "IR2", "IR3"])
dict_encoder["LandContour"] = create_fit_encoder(["Lvl", "Bnk", "HLS", "Low"])
dict_encoder["LotConfig"] = create_fit_encoder(["Inside", "Corner", "CulDSac", "FR2", "FR3"])
dict_encoder["LandSlope"] = create_fit_encoder(["Gtl", "Mod", "Sev"])
dict_encoder[""] = create_fit_encoder([])
dict_encoder[""] = create_fit_encoder([])
dict_encoder[""] = create_fit_encoder([])
dict_encoder[""] = create_fit_encoder([])
dict_encoder[""] = create_fit_encoder([])
dict_encoder[""] = create_fit_encoder([])

