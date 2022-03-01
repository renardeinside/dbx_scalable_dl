from dbx_scalable_dl.common import Job


class DataLoaderTask(Job):
    def launch(self):
        self.logger.info("Starting the data loader job")
        self.logger.info("Data loading successfully finished")


if __name__ == '__main__':
    DataLoaderTask().launch()
