from data.loader import FileIO


class SELFRec(object):
    def __init__(self, config,i):
        self.i = i
        self.config = config
        path1 = f"./dataset/mydata/train_{i}.txt"
        path2 = f"./dataset/mydata/ran_test_{i}.txt"
        self.training_data = FileIO.load_data_set(path1, config['model.type'])
        self.test_data = FileIO.load_data_set(path2, config['model.type'])

        self.kwargs = {}
        if config.contain('social.data'):
            social_data = FileIO.load_social_data(self.config['social.data'])
            self.kwargs['social.data'] = social_data
        # if config.contains('feature.data'):
        #     self.social_data = FileIO.loadFeature(config,self.config['feature.data'])
        print('Reading data and preprocessing...')

    def execute(self):
        # import the model module
        import_str = 'from model.'+ self.config['model.type'] +'.' + self.config['model.name'] + ' import ' + self.config['model.name']
        exec(import_str)
        recommender = self.config['model.name'] + '(self.config,self.training_data,self.test_data,self.i,**self.kwargs)'
        eval(recommender).execute()
