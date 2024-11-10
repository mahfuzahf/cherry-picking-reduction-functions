from tqdm import tqdm
import functions as f


# class to represent a vanilla rainbow table
class RainbowTable:
    def __init__(self, N, mt_target, maximality_factor, t, m0):
        self.N = N
        self.mt_target = mt_target
        self.maximality_factor = maximality_factor
        self.t = t
        self.m0 = m0
        self.table = self.init_rainbow_table()
        self.init_rainbow_table()

    # function to initialise a rainbow table
    def init_rainbow_table(self):
        # initialize the table as a set with all m0 start points
        table = {}
        # start from 0 and go up to m0
        for i in range(self.m0):
            # set all the start points to i
            x = i
            table[i] = x

        return table

    def build_rainbow_table(self):
        # for each column in the table
        for i in tqdm(range(self.t)):
            # store list of keys to remove - as can't remove while looping through dictionary
            to_remove = []

            # want to hash and reduce values and remove duplicates
            # iterate through the set
            # store values we have seen before in a set and search this set when adding new values
            seen = set()
            seenSize = 0

            # for each row in the table
            for a, b in self.table.items():
                # hash and reduce each the current value
                x = f.r(f.H(b), i, 0, t, N)
                # add this value to seen set
                seen.add(x)

                # if the seen set did not increase in size, we have a duplicate
                if seenSize == len(seen):
                    # add to list of items to remove
                    to_remove.append(a)
                else:
                    # update length of seen set
                    seenSize = len(seen)
                    # update the dictionary
                    self.table[a] = x

            # remove duplicates
            for key in to_remove:
                self.table.pop(key)


# rainbow table parameters
N = 2 ** 16  # number of possible plaintexts
mt_target = N ** (2 / 3)  # target mt
maximality_factor = 0.9
t = 100
m0 = round(mt_target / (1 - maximality_factor))

# create a rainbow table
rainbow_table = RainbowTable(N, mt_target, maximality_factor, t, m0)

# build the rainbow table
rainbow_table.build_rainbow_table()

print(len(rainbow_table.table))
