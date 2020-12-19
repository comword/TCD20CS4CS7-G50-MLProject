import sqlite3

from sklearn.model_selection import train_test_split

conn = sqlite3.connect('data/WDIProd.sqlite')


def getMaxMinForData(row):
    return r'SELECT max("{}") as maxvalue, min("{}") as minValue, avg("{}") as avgvalue FROM "data"'.format(row, row,
                                                                                                            row)


class DataSelector:
    def __init__(self, yearDelta, metrics, note=""):
        self.yearDelta = yearDelta
        self.metrics = metrics
        self.note = note


class QueryConstructor:
    def __init__(self, dataSelector):
        self.dataSelector = dataSelector

    def formartTableName(self, table, row):
        return r'"{}"."{}"'.format(table, row)

    def formartName(self, table):
        return r'"{}"'.format(table)

    def constructForYear(self, year):
        output = ""

        # SELECT PART FIRST
        output += r'SELECT '

        for selectedData in self.dataSelector.metrics:
            output += self.formartTableName("time1", selectedData)
            output += " as "
            output += self.formartName("p." + selectedData)
            output += " , "

        output += " "

        # Country Name as N row
        output += self.formartTableName("origins", "CountryCode")
        output += ", "

        # T+0 GDP as N+1 row
        output += self.formartTableName("time2", "NY.GDP.PCAP.PP.KD")
        output += " as gdpf ,"

        # T+yY GDP as N+2 row
        output += self.formartTableName("time1", "NY.GDP.PCAP.PP.KD")
        output += " as gdpp "

        year2 = year
        year1 = year - self.dataSelector.yearDelta

        data = r' from (SELECT DISTINCT "origin"."CountryCode" from "data" as "origin") as "origins" LEFT JOIN "data" as "time1" ON "time1"."CountryCode" == "origins"."CountryCode" AND "time1"."Year" == {} LEFT JOIN "data" as "time2" ON "time2"."CountryCode" == "origins"."CountryCode" AND "time2"."Year" == {} WHERE gdpf is not NULL AND gdpp is not NULL'.format(
            year1, year2)

        output += data

        return output


def Normalize(meanv, minmaxdiffv):
    """
    :param a: Array
    :return: Array
    """
    mean = meanv
    maxmindiff = minmaxdiffv

    def apply(v):
        if v is None:
            return 0
        return (v - mean) / maxmindiff

    def apply_inversed(v):
        return v * maxmindiff + mean

    return apply, apply_inversed


class DataFromSelector:
    def __init__(self, dataSelector, yearmin, years, conn):
        self.dataSelector = dataSelector
        self.yearmin = yearmin
        self.years = years  # The window of years

        self.qc = QueryConstructor(self.dataSelector)
        self.conn = conn

        normalF = []
        for mat in self.dataSelector.metrics:
            nf = self.getNormalizeFunction(mat)[0]
            normalF.append(nf)

        self.normalF = normalF

    def getNormalizeFunction(self, metric):
        sql = getMaxMinForData(metric)

        maxmin = self.conn.execute(sql).fetchone()

        maxv = maxmin[0]
        minv = maxmin[1]

        avgv = maxmin[2]

        return Normalize(avgv, maxv - minv)

    def getProsperityIndex(self, gdp1, gdp2):
        if gdp2 - gdp1 > 0:
            return pow((gdp2 - gdp1) / gdp1, (1 / self.years))
        else:
            return - pow((gdp1 - gdp2) / gdp1, (1 / self.years))

    def getXfromRow(self, row):
        rawData = row[:-3]
        normalF = self.normalF

        data = []
        for item in range(len(rawData)):
            data.append(normalF[item](rawData[item]))
        return data

    def getYfromRow(self, row):
        return self.getProsperityIndex(row[-2], row[-1])

        # return (row[-2],row[-1])

    def constructOneYear(self, year, X, Y):
        sql = self.qc.constructForYear(year)
        res = self.conn.execute(sql)
        for data in res:
            X.append(self.getXfromRow(data))
            Y.append(self.getYfromRow(data))

    def constructAll(self):
        X = []
        Y = []

        for ye in range(self.yearmin, self.yearmin + self.years):
            self.constructOneYear(ye, X, Y)

        return (X, Y)


class CombinedDataSelector:
    def __init__(self, daArray):
        self.daArray = daArray

    def Construct(self):
        for itemi in range(self.daArray):
            cloned = list(self.daArray)
            cloned.remove(itemi)
            selected = []
            for v in cloned:
                selected.extend(v.metrics)
            yield (selected, itemi)


def TrainingPipeline(daArray, conn, model):
    cds = CombinedDataSelector(daArray)
    for s in cds.Construct():
        (selected, itemi) = s
        ds = DataSelector(5, selected)
        dfs = DataFromSelector(ds, 1980, 25, conn)
        datas = dfs.constructAll()
        (X, y) = datas
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
        modelc = model.getNewConsolitedModel()
        modelc.fit(X_train, y_train)
        Score = modelc.evaluate(X_test, y_test)[0]
        print(itemi.note, Score)
