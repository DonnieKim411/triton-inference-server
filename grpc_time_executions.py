import argparse
import datetime

import tritongrpcclient
from tritonclientutils import InferenceServerException

# set logger
import logging
logging.basicConfig(level=logging.DEBUG)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-v',
                        '--verbose',
                        action="store_true",
                        required=False,
                        default=False,
                        help='Enable verbose output')
    parser.add_argument('-u',
                        '--url',
                        type=str,
                        required=False,
                        default='localhost:8001',
                        help='Inference server URL. Default is localhost:8001.')

    FLAGS = parser.parse_args()

    # measure time to create inference context
    start = datetime.datetime.now()
    try:
        triton_client = tritongrpcclient.InferenceServerClient(url=FLAGS.url,
                                                         verbose=FLAGS.verbose)
    except Exception as e:
        print("context creation failed: " + str(e))
        sys.exit()

    end = datetime.datetime.now()
    logging.info("Task 0: inferance context creation duration: {}".format(end-start))
    
    model_name = 'simple'

    # Health

    # measure time to check whether the server is live or not
    start = datetime.datetime.now()
    if not triton_client.is_server_live(headers={'test': '1', 'dummy': '2'}):
        print("FAILED : is_server_live")
        sys.exit(1)

    end = datetime.datetime.now()
    logging.info("Task 1: checking whether the sever is live duration: {}".format(end-start))
    
    # measure time to check whether the server is live or not
    start = datetime.datetime.now()
    if not triton_client.is_server_ready():
        print("FAILED : is_server_ready")
        sys.exit(1)

    end = datetime.datetime.now()
    logging.info("Task 2: checking whether the sever is ready duration: {}".format(end-start))

    # measure time to check whether the model is ready or not
    start = datetime.datetime.now()
    if not triton_client.is_model_ready(model_name):
        print("FAILED : is_model_ready")
        sys.exit(1)

    end = datetime.datetime.now()
    logging.info("Task 3: checking whether the model is ready duration: {}".format(end-start))


    # Server Metadata
    start = datetime.datetime.now()
    metadata = triton_client.get_server_metadata()
    if not (metadata.name == 'triton'):
        print("FAILED : get_server_metadata")
        sys.exit(1)
    print(metadata)

    end = datetime.datetime.now()
    logging.info("Task 4: Obtaining server meta data duration: {}".format(end-start))

    # Model Metadata
    start = datetime.datetime.now()
    metadata = triton_client.get_model_metadata(model_name,
                                                headers={
                                                    'test': '1',
                                                    'dummy': '2'
                                                })
    if not (metadata.name == model_name):
        print("FAILED : get_model_metadata")
        sys.exit(1)
    print(metadata)

    end = datetime.datetime.now()
    logging.info("Task 5: Obtaining Model meta data duration: {}".format(end-start))

    # Passing incorrect model name
    try:
        metadata = triton_client.get_model_metadata("wrong_model_name")
    except InferenceServerException as ex:
        if "Request for unknown model" not in ex.message():
            print("FAILED : get_model_metadata wrong_model_name")
            print("Got: {}".format(ex.message()))
            sys.exit(1)
    else:
        print("FAILED : get_model_metadata wrong_model_name")
        sys.exit(1)

    # Get Model Configuration
    start = datetime.datetime.now()
    config = triton_client.get_model_config(model_name)
    if not (config.config.name == model_name):
        print("FAILED: get_model_config")
        sys.exit(1)
    end = datetime.datetime.now()
    logging.info("Task 6: Obtaining model configuration duration: {}".format(end-start))
