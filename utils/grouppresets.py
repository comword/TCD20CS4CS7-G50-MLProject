from .dataloader import DataSelector


def Groups(year):
    group = []
    group.append(DataSelector(year, ["NY.GDP.PCAP.PP.KD"], "Size of the economy (GDP per capita)"))
    group.append(DataSelector(year, ["SI.POV.LMIC.GP"], "Poverty rates at international poverty lines"))
    group.append(DataSelector(year,
                              ["SI.DST.10TH.10", "SI.DST.05TH.20", "SI.DST.FRST.10", "SI.DST.FRST.20", "SI.DST.50MD",
                               "SI.POV.GINI"], "Distribution of income or consumption"))
    group.append(DataSelector(year, ["SL.TLF.0714.ZS", "SL.EMP.1524.SP.ZS", "SL.TLF.ADVN.ZS", "SL.TLF.BASC.ZS",
                                     "SL.TLF.INTM.ZS"], "Labor force structure"))
    group.append(DataSelector(year, ["SL.IND.EMPL.ZS", "SL.ISV.IFRM.ZS", "SL.EMP.WORK.ZS", "SL.AGR.EMPL.ZS"],
                              "Employment by sector"))
    group.append(DataSelector(year, ["SL.UEM.1524.ZS", "SL.UEM.TOTL.ZS"], "Unemployment"))
    group.append(DataSelector(year, ["SP.DYN.LE00.IN", "SH.XPD.OOPC.CH.ZS", "SH.XPD.CHEX.PP.CD"], "Health systems"))
    group.append(DataSelector(year, ["EG.EGY.PRIM.PP.KD", "EG.ELC.ACCS.ZS", "EG.ELC.COAL.ZS", "EN.ATM.CO2E.PP.GD"],
                              "Energy dependency, efficiency and carbon dioxide emissions"))
    group.append(DataSelector(year, ["BX.GSR.MRCH.CD", "TX.VAL.AGRI.ZS.UN", "TX.VAL.FOOD.ZS.UN", "TX.VAL.FUEL.ZS.UN"],
                              "Structure of merchandise exports"))
    group.append(DataSelector(year, ["TM.VAL.AGRI.ZS.UN", "TM.VAL.FOOD.ZS.UN", "TM.VAL.FUEL.ZS.UN"],
                              "Structure of merchandise imports"))
    group.append(DataSelector(year, ["BX.GSR.NFSV.CD", "NE.EXP.GNFS.KD", "TX.VAL.ICTG.ZS.UN", "TX.VAL.INSF.ZS.WT"],
                              "Structure of service exports"))
    group.append(DataSelector(year, ["NE.IMP.GNFS.KD", "TM.VAL.ICTG.ZS.UN", "TM.VAL.MANF.ZS.UN"],
                              "Structure of service imports"))
    group.append(
        DataSelector(year, ["IC.FRM.DURS", "IC.REG.PROC", "IC.REG.COST.PC.ZS", "IC.LGL.DURS", "IC.GOV.DURS.ZS"],
                     "Business environment: Doing Business indicators"))
    group.append(DataSelector(year, ["FB.CBK.BRCH.P5", "FB.CBK.BRWR.P3", "FR.INR.RISK"],
                              "Financial access, stability and efficiency"))
    group.append(DataSelector(year, ["GC.TAX.EXPT.CN", "GC.TAX.GSRV.VA.ZS", "GC.TAX.OTHR.RV.ZS", "GC.TAX.YPKG.RV.ZS"],
                              "Tax policies"))

    return group



