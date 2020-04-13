class Parser: 
    #
    # REPLACE THE ORIGINAL create_dataset FUNCTION WITH THIS ONE
    #
    def create_dataset(self, source, train=False) :
        """
        Creates a dataset from all parser configurations encountered
        during parsing of the training dataset.
        (Not used in assignment 1).
        """
        ds = Dataset()
        with open(filename) as source:
            for w,tags,tree,relations in self.trees(source) : 
                i, stack, pred_tree = 0, [], [0]*len(tree) # Input configuration
                m = self.compute_correct_move(i,stack,pred_tree,tree)
                while m != None :
                    ds.add_datapoint(w, tags, i, stack, m, train)
                    i,stack,pred_tree = self.move(i,stack,pred_tree,m)
                    m = self.compute_correct_move(i,stack,pred_tree,tree)
        return ds

