# ermmmm i dont think this is needed anymore


import functions as f


# A set where elements have two parts: a key and a value
# the key is the order the element was added to the set
class TableSet:
    def __init__(self):
        self.set = set()
        self.keyCount = 0

    def initialize(self, t):
        # insert t elements with key and value as keyCount
        for i in range(t):
            self.add_init(i)

    # add a value to the set with its key as a tuple
    def add_init(self, value):
        toAdd = (self.keyCount, value)
        self.set.add(toAdd)
        # increment the key count
        self.keyCount += 1

    def update_value(self, index, oldvalue, newvalue):
        # remove the tuple
        self.set.remove((index, oldvalue))
        # add the new tuple
        self.set.add((index, newvalue))

    def build_rainbow_table(self, t, N):
        # for each column in the table
        for i in range(t):
            # want to hash and reduce values and remove duplicates
            # iterate through the set
            # store values we have seen before in a set and search this set when adding new values
            seen = set()
            seenSize = 0
            # for each row in the table
            for item in self.set:
                # hash and reduce each the current value
                x = f.r(f.H(item[1]), i, 0, t, N)
                # add this value to seen set
                seen.add(x)
                # if the seen set did not increase in size, we have a duplicate
                if seenSize == len(seen):
                    # remove the duplicate
                    self.set.remove(item)
                else:
                    seenSize =+ 1
                    self.set.update(item, x)
                    # update the tuple in the set


testy = TableSet()
testy.initialize(4)
print(testy.set)

newval = [13, 24, 13, 2]

seen = set()
seenSize = 0
count = 0
# for each row in the table
for item in testy.set:
    # store which key to remove then remove at the end
    toremove = []
    # x = iterated value in newval
    x = newval[count]
    # add this value to seen set
    seen.add(x)
    # if the seen set did not increase in size, we have a duplicate
    if seenSize == len(seen):
        # remove the duplicate
        testy.set.remove(item)
    else:
        seenSize = len(seen)
        testy.update_value(item[0], item[1], x)
        # update the tuple in the set

    count += 1

print(testy.set)