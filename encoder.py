from sklearn.preprocessing import LabelEncoder
import numpy as np


class Encoder(LabelEncoder):
    def fit_more(self, more_data):
        print("testTestTest")
        le = LabelEncoder()
        le.fit(more_data)
        for element in le.classes_:
            print(le.classes_)
            if element not in self.classes_:
                self.classes_ = np.append(self.classes_, [element])
                print(self.classes_)

"""
def create_fit_encoder(para):
    le = LabelEncoder()
    le.fit(para)
    return le


def get_encoder():
    dict_encoder = dict
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
    return dict_encoder
"""
if __name__ == '__main__':
    le = Encoder()
    le.fit([0, 1, 2, 3])
    le.fit_more([4])
    print(le.classes_)

