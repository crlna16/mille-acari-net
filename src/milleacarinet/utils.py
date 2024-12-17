
import logging

def create_logger(log):
    '''Adapt logger to our needs.
    
    Args:
      log (Logger): Logger instance.
      
    Returns: 
      Logger.
    '''
    log.setLevel(logging.INFO)  # Set the logging level to INFO
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    formatter = logging.Formatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    console_handler.setFormatter(formatter)
    log.addHandler(console_handler)
    return log