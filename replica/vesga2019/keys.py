# State space indexing
N_State = 12

I_U = 0
I_Lat = 1

I_Asym = 2
I_Sym = 3
I_DPub = 4
I_DPri = 5
I_E = 6

I_TPub = 7
I_TPri = 8

I_RLow = 9
I_RHigh = 10
I_RStab = 11

I_Act = [I_Asym, I_Sym]
I_D = [I_DPub, I_DPri]
I_T = [I_TPub, I_TPri]
I_Infectious = I_Act + I_D + [I_E]
I_Prevalent = I_Infectious + I_T
I_LTBI = [I_Lat, I_RLow, I_RHigh, I_RStab]
