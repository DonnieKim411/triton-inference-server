
# """A client that talks to triton server.

# Typical usage example:

#     python http_call_tester.py --batch_size=100
# """

import sys, argparse, grpc, datetime
import numpy as np
import queue

import tritongrpcclient.model_config_pb2 as mc
import tritonhttpclient
from tritonclientutils import triton_to_np_dtype
from tritonclientutils import InferenceServerException

# set logger
import logging
logging.basicConfig(level=logging.DEBUG)


class UserData:
    def __init__(self):
        self._completed_requests = queue.Queue()


# Callback function used for async_stream_infer()
def completion_callback(user_data, result, error):
    # passing error raise and handling out
    user_data._completed_requests.put((result, error))


def parse_model_http(model_metadata, model_config):
    """
    Check the configuration of a model to make sure it meets the
    requirements for an image classification network (as expected by
    this client)
    """
    if len(model_metadata['inputs']) != 1:
        raise Exception("expecting 1 input, got {}".format(
            len(model_metadata['inputs'])))
    if len(model_metadata['outputs']) != 1:
        raise Exception("expecting 1 output, got {}".format(
            len(model_metadata['outputs'])))

    if len(model_config['input']) != 1:
        raise Exception(
            "expecting 1 input in model configuration, got {}".format(
                len(model_config['input'])))

    input_metadata = model_metadata['inputs'][0]
    output_metadata = model_metadata['outputs'][0]

    max_batch_size = 0
    if 'max_batch_size' in model_config:
        max_batch_size = model_config['max_batch_size']

    if output_metadata['datatype'] != "FP32":
        raise Exception("expecting output datatype to be FP32, model '" +
                        model_metadata['name'] + "' output type is " +
                        output_metadata['datatype'])

    # Output is expected to be a vector. But allow any number of
    # dimensions as long as all but 1 is size 1 (e.g. { 10 }, { 1, 10
    # }, { 10, 1, 1 } are all ok). Ignore the batch dimension if there
    # is one.
    output_batch_dim = (max_batch_size > 0)
    non_one_cnt = 0
    for dim in output_metadata['shape']:
        if output_batch_dim:
            output_batch_dim = False
        elif dim > 1:
            non_one_cnt += 1
            if non_one_cnt > 1:
                raise Exception("expecting model output to be a vector")

    # Model input must have 3 dims (not counting the batch dimension),
    # either CHW or HWC
    input_batch_dim = (max_batch_size > 0)
    expected_input_dims = 3 + (1 if input_batch_dim else 0)
    if len(input_metadata['shape']) != expected_input_dims:
        raise Exception(
            "expecting input to have {} dimensions, model '{}' input has {}".
            format(expected_input_dims, model_metadata['name'],
                   len(input_metadata['shape'])))

    return (max_batch_size, input_metadata['name'], output_metadata['name'], input_metadata['datatype'])


def requestGenerator(batched_image_data, input_name, output_name, dtype, flags):

    # Set the input data
    inputs = []
    inputs.append(
        tritonhttpclient.InferInput(input_name, 
                                    batched_image_data.shape,
                                    dtype)
        )
    inputs[0].set_data_from_numpy(batched_image_data, binary_data=False)

    # Set the output data
    outputs = []
    outputs.append(
        tritonhttpclient.InferRequestedOutput(output_name, binary_data=False)
        )

    # print(batched_image_data.shape)

    # infer_input = tritonhttpclient.InferInput(input_name, 
    #                                 batched_image_data.shape,
    #                                 dtype)
    # infer_input.set_data_from_numpy(batched_image_data, binary_data=False)
    # infer_output = tritonhttpclient.InferRequestedOutput(output_name, binary_data=False)

    yield inputs, outputs
    # yield [infer_input], [infer_output]


class ToyClientHttp():
    """
    Http Toy Client
    """

    def __init__(self, flags):

        try:
            self.triton_client = tritonhttpclient.InferenceServerClient(url=flags.url,
                                                            verbose=flags.verbose)
        except Exception as e:
            print("context creation failed: " + str(e))
            sys.exit()

        # Make sure the model matches our requirements, and get some
        # properties of the model that we need for preprocessing
        try:
            model_metadata = self.triton_client.get_model_metadata(
                model_name=flags.model_name, model_version=flags.model_version)
        except InferenceServerException as e:
            print("failed to retrieve the metadata: " + str(e))
            sys.exit(1)

        try:
            model_config = self.triton_client.get_model_config(
                model_name=flags.model_name, model_version=flags.model_version)
        except InferenceServerException as e:
            print("failed to retrieve the config: " + str(e))
            sys.exit(1)

        max_batch_size, input_name, output_name, input_dtype = parse_model_http(
            model_metadata, model_config)

        self.max_batch_size = max_batch_size
        self.input_name = input_name
        self.output_name = output_name
        self.input_dtype = input_dtype

    def infer(self, input_data, output_data, flags, sent_count):
        """
        infer
        """

        # infer asynchronously
        return self.triton_client.async_infer( 
                                            flags.model_name, 
                                            input_data,
                                            request_id=str(sent_count),
                                            model_version=flags.model_version,
                                            outputs=output_data)

def run(flags):

    random_img = np.random.uniform(low=0.0, high=1.0, size=(flags.batch_size,150,40,1)).astype(np.float32)

    # initialize the toy client http
    triton_client = ToyClientHttp(flags)
    
    # client = ToyClient('auroraaidev:8500')
    responses = []

    # Holds the handles to the ongoing HTTP async requests.
    async_requests = []

    sent_count = 0

    # measure time it takes
    start = datetime.datetime.now()

    # # describe input tensor for an inference request
    # infer_input = tritonhttpclient.InferInput(triton_client.input_name, random_img.shape, triton_client.input_dtype)
    # infer_input.set_data_from_numpy(random_img, binary_data=False)

    # # describe a requested output tensor for an inference request
    # infer_output = tritonhttpclient.InferRequestedOutput(triton_client.output_name, binary_data=False)

    # try:
    #     inference = triton_client.infer(infer_input, infer_output, flags, sent_count)

    # except InferenceServerException as e:
    #     print("inference failed: " + str(e))
    #     sys.exit(1)
    

        # result = inference.get_result()
        # responses.append(result.as_numpy(triton_client.output_name))



    # Send request over batch
    try:
        for inputs, outputs in requestGenerator(random_img, 
                                                triton_client.input_name, 
                                                triton_client.output_name, 
                                                triton_client.input_dtype, 
                                                flags):
            print(inputs, outputs)
            async_requests.append(triton_client.infer(inputs, outputs, flags, sent_count))
            print(sent_count)
            sent_count += 1

    except InferenceServerException as e:
        print("inference failed: " + str(e))
        sys.exit(1)

    # Collect results from the ongoing async requests
    # for HTTP Async requests.

    print("# of requests: {}".format(async_requests.__len__()))

    for async_request in async_requests:

        result = async_request.get_result()
        responses.append(result.as_numpy(triton_client.output_name))

    end = datetime.datetime.now()

    # print(responses, responses[0].shape)

    logging.info("inferance duration time: {}".format(end-start))


def main(flags):

    run(flags)

if __name__ == '__main__':
    text = "A simple testing program. It generates random images in shape of (n x 150 x 40 x 1)\
        where n is the number of batch size the user provides. The script will connect to the host\
        where the model is uploaded, sends the generated random images, then print out the probabilities.\
        The user then must fetch the print statement and group them in 3 where each index represents each class.\
        "
    parser = argparse.ArgumentParser(description=text)
    parser.add_argument("--batch_size", default=1, help="batch size of the data. Default 1", type=int)
    parser.add_argument("--url", help="http url with port", type=str)
    parser.add_argument("--verbose", default=False, type=bool)
    parser.add_argument("--model_name", default="grex_model", help="model name", type=str)
    parser.add_argument("--model_version", default="1", help="version of the model", type=str)
    parser.add_argument('--print_prediction', dest='print_prediction', action='store_true', help="print out the prediction")
    parser.add_argument('--no-print_prediction', dest='print_prediction', action='store_false', help="DO NOT print out the prediction")
    parser.set_defaults(print_prediction=True)
    parser.add_argument("--job_number", default=1, help="job number. Only use for debugging purpose. Default 1")
    # parser.add_argument('--save_logging', dest='save_logging', action='store_true', help="save logging")
    # parser.add_argument('--no-save_logging', dest='save_logging', action='store_false', help="DO NOT save logging")
    parser.set_defaults(save_logging=False)
    
    flags = parser.parse_args()

    main(flags)

