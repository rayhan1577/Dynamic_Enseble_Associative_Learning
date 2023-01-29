from transaction import Transaction
import util
from config import Configuration as config
import random
random.seed(1)

class TransactionalDatabase:
    ''' Represents a transactional database
    How to use this class?

    '''
    _comb = util.my_comb
    _fact = util.my_factorial

    def __init__(self, file_name, index = None, separator=',', use_corollary_1=False):
        ''' Makes a transactional database
        by calling the retrieve_transactions function.
        
        Args:
        file_name: Name of the file containing transactions
        index: index of the class label which is valid
            valid for all transactions(0-based)
            The default is the last value in each line.
        separator: the separator between items (and the label) in the 
        database file. The default value is ','.
        '''
        if separator is None:
            separator = ','
        # A dictionary of transactions objects in the database(key is the id 
        # of the transaction)
        self._transactions = dict()
        # For each item, we keep track of the transaction ids it belongs to. 
        # (in a set)
        self._items_dict   = dict()
        # A set of labels in the databse
        self._labels_dict   = dict()
        self.__retrieve_transactions(file_name, index, separator)
        if use_corollary_1:
            threshold = self.__find_threshold()
            print('threshold for support is:', threshold)
            items_to_be_removed = self.__get_non_frequent_items(threshold)
            print(len(items_to_be_removed), 'items are going to be removed.')
            self.__remove_non_frequent_items(items_to_be_removed)

    def __find_threshold(self):
        d_size = self.get_database_size()
        for i in range(1,int(self.get_database_size()/2)):
            min_val = (TransactionalDatabase._fact(i)*TransactionalDatabase._fact(d_size - i)) / TransactionalDatabase._fact(d_size)
            print(i, min_val)
            if not (min_val>=config.ALPHA):
                return i-1
        return 0
    def __get_non_frequent_items(self, threshold):
        to_be_removed = []
        for item, transaction_ids in self._items_dict.items():
            support = len(transaction_ids)
            if support<=threshold:
                to_be_removed.append(item)
        return to_be_removed

    def __remove_non_frequent_items(self, to_be_removed):
        for item in to_be_removed:
            if item in self._items_dict:
                del self._items_dict[item]
        for _,transaction in self._transactions.items():
            transaction.remove_any(to_be_removed)


    def __retrieve_transactions(self, file_name, global_label_index, separator):
        ''' Reads a file containing set of transactions (one per line)
        In each line, there should be items and the class label.
        The function makes transaction objects and returns them together 
        with set of items and set of labels.

        Args: 
        file_name: name of the file containing transaction database.
        global_label_index: index of the class label that is valid for all tokens
                            It is recommended to set the class label as the last
                            token in each line, otherwise, you need to make sure
                            that the given index is valid in all lines.
        separator: the separator between items (and the label) in the database file
            The default value is ','.
        '''
        with open(file_name, 'r') as fin:
            for index, line in enumerate(fin):
                tokens = line.strip().split(separator)

                # an empty line or a line with only one value on it in the file:
                if len(tokens)==0:
                    print('warning, an empty line on the input file:', index)
                    continue
                if len(tokens)==1:
                    print('warning, a line with only one value on the input file:', index)
                    continue

                if global_label_index==None:
                    label_index = len(tokens)-1
                transaction = Transaction(tokens[:label_index] + tokens[label_index+1:], 
                                                                    tokens[label_index], 
                                                                                  index)
                self._transactions[index] = transaction
                #self._labels_dict.add(tokens[label_index])

                # increase the count of the class label.
                if tokens[label_index] in self._labels_dict:
                    self._labels_dict[tokens[label_index]] += 1
                else:
                    self._labels_dict[tokens[label_index]]  = 1
                # increase the count of each item.
                for item in transaction.get_items():
                    if item in self._items_dict:
                        self._items_dict[item].add(index)
                    else:
                        self._items_dict[item] = set([index])

    def get_database_size(self):
        ''' Returns the size of the database 
        (how many transactions it has)

        Returns:
        An integer showing the size of the transactional database.
        '''
        return len(self._transactions)


    def get_sorted_items_list(self, descending=False):
        ''' Returns the list of items that is sorted based on the
        support value for each item.

        Args:
        descending: Whether the list should be sorted in descending order
            or not (default: False)

        Returns:
        A list containing item names, the list is ordered (default:descending)
        '''
        #d = self._items_dict.items()
        #return sorted([x for x,y])
        temp0 = {x[0]:len(x[1]) for x in self._items_dict.items()}
        temp = sorted(list(temp0.items()), 
                        key= lambda x: (len(self._items_dict[x[0]]),float(x[0])), 
                                                            reverse= descending) #int
        #print('sorted items based on count in databse:', temp)
        #print(temp0)
        return [x[0] for x in temp]

    def get_transactions(self):
        ''' This function yields transactions in the database.

        Return:
        It yields a transaction object.
        '''
        for transaction_id, transaction in self._transactions.items():
            yield transaction

    def get_label_support(self, label):
        ''' Given a class label, this function returns its support value.

        Args:
        label: a class label

        Returns:
        the corresponding support count for this label.
        '''
        if label not in self._labels_dict:
            return 0
        return self._labels_dict[label]

    def get_all_labels(self):
        return list(self._labels_dict.keys())

    def __str__(self):
        '''Returns a string corresponding to the transactional database
        Each line consists of information of one of transaction.
        They are randomly ordered.

        Returns:
        A string corresponding to information of the transactional database.
        '''
        labels = 'labels and their count- '
        for label, support in self._labels_dict.items():
            labels += str(label) + ':' + str(support) + '\t'
        labels += '\n'
        description = '\nid: items --> class\n'
        rets = []
        for id_, transaction in self._transactions.items():
            rets.append(str(id_) + ' : ' + str(transaction))
        return labels + description + '\n'.join(rets) + '\n'


def test_1(filename):
    d = TransactionalDatabase(filename)
    print('_items_dict',   d._items_dict)
    print('_labels_dict',   d._labels_dict)
    print('_transactions', d._transactions)
    print('get_sorted_items_list', d.get_sorted_items_list())
    print('obj:', d)
if __name__ == '__main__':
    test_1('test.input')