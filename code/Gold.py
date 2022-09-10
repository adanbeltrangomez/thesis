import csv
import Utils_Pln
import Main
import pandas as pd


class Gold:

    def __init__(self):
        '''

        '''

        # self.oUtilsPln = Utils_Pln.UtilPln()

        # self._NameFileNorms = self.oUtilsPln.Read_Config('FilePath_Norms')[0]

        self.norms = {}

    def evaluate(self, community, section, id):
        self.norms = self.readGold(id)

        for n in self.norms:
            n.process()



    def printResults(self):
        Main.Norm.writeCSV(self, self.norms)
        return

    def printGold(self, corpus, section):

        titles = ["Corpus", "Id", "Text", "g_s", "g_v", "g_o", "g_structure", "g_m", "g_p1", "g_p2", "g_a", "g_c",
                  "comments", "url"]
        lines = []
        lines.append(titles)

        for n in self.norms:
            # if n[8]  not in corpus or n[9] not in section:
            #    continue

            # name    = n[8]
            id = n.ID
            text = n.TEXT
            g_s = n.G_SUBJECT
            g_PClause = n.G_P_CLAUSE
            g_QClause = n.G_Q_CLAUSE
            g_v = n.G_VERBS
            g_o = n.G_OBJECT
            g_m = n.G_MODALITY
            g_p1 = n.G_PERSON1
            g_p2 = n.G_PERSON2
            g_c = n.G_P_CLAUSE
            g_a = n.G_ACTION

            lines.append([id, text, g_PClause, g_QClause, g_v, g_o, g_m, g_p1, g_p2, g_a, g_c])

        with open('Gold.csv', 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile, delimiter=';', quoting=csv.QUOTE_MINIMAL)

            for line in lines:
                writer.writerow(line)

    def readGold(self, id_norm):

        df = pd.read_excel("GoldStandard.xlsx", sheet_name='Gold')
        df = df.fillna("")

        self.norms = []  # {}

        for index, row in df.iterrows():
            oNorm = Main.Norm()

            oNorm.ID = row["Id"]
            if id_norm != "" and oNorm.ID != id_norm:
                continue

            oNorm.TEXT = row["Text"]
            oNorm.G_P_CLAUSE = row["P_Clause"].split("::")
            oNorm.G_Q_CLAUSE = row["Q_Clause"].split("::")
            oNorm.G_SUBJECT = row["Subject"].split("::")
            oNorm.G_OBJECT = row["Object"].split("::")
            oNorm.G_VERBS = row["Verb"].split("::")
            oNorm.G_MODALITY = row["Modality"].split("::")
            oNorm.G_PERSON1 = row["Person1"].split("::")
            oNorm.G_PERSON2 = row["Person2"].split("::")
            oNorm.G_P_VERB = row["P_Verb"].split("::")
            oNorm.G_P_VERB_TENSE = row["P_Verb_Tense"].split("::")
            oNorm.G_Q_VERB = row["Q_Verb"].split("::")
            oNorm.G_Q_VERB_TENSE = row["Q_Verb_Tense"].split("::")
            oNorm.G_STRUCTURE = row["Structure"]
            oNorm.G_ACTION = row["Action"].split("::")
            oNorm.G_VOICE = row["Voice"].split("::")
            oNorm.G_P_VERB_Tense_Q_VERB_TENSE = row["P_Verb_Tense+Q_Verb_Tense"]
            oNorm.G_CANONICAL = row["Canonical"]

            print("Procesando:\n", oNorm.ID)

            self.norms.append(oNorm)
        return self.norms


oGold = Gold()

# oGold.evaluate("GOLD-STANDARD-FACEBOOK","NORM","")
# oGold.print("GOLD-STANDARD-FACEBOOK","NORM")

oGold.evaluate("GOLD-STANDARD-BYU-BNC", "NORM", "")
oGold.printResults()

#oGold.printGold("GOLD-STANDARD-BYU-BNC", "NORM")
