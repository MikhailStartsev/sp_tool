#!/usr/bin/env python
from collections import OrderedDict
import arff
import warnings
import numpy as np
import numpy.lib.recfunctions as rfn


class ArffHelper(object):
    """
    The class is based on general arff handler with an extra keyword %@METADATA
    (they are comment lines in the description, i.e. lines *before* the
    `relation` keyword).

    Metadata fields contains metadata names and related values (separated by space characters).

    - Lines starting with "%" are comments, except for line starting with %@METADATA
    - Lines starting with "@" that is followed by a word (without space), are considered
      keywords. The available keywords are the following:
        @RELATION: a string with the name of the data set.
        @ATTRIBUTES: a list of attributes representing names of data columns
                     followed by the types of data. The available data types
                     are 'NUMERIC', 'REAL', 'INTEGER' or a list of string.
        @DESCRIPTION: a string with the description of the data set.
        @DATA: a list of data instances. The data should follow the order that
               the attributes were presented.
    - Metadata ('%@METADATA <KEY> <VALUE>' lines) can have any keys, but the following ones are subsequently used
      during eye movement classification:
        (1) "width_px", horizontal resolution of the video during recording, in pixels
        (2) "height_px, vertical resolution of the video during recording, in pixels
        (3) "distance_mm", distance of the observer's eyes from the monitor during recording, in millimeters
        (4) "width_mm", horizontal physical dimensions of the video surface during recording, in millimeters
        (5) "height_mm", vertical physical dimensions of the video surface during recording, in millimeters

    The metadata values are extracted from the description section (comments before @RELATION) of the arff file,
    and are placed in the 'metadata' key of the arff object (which is a dictionary itself) during the loading of the
    arff file.

    During the dumping of the arff object, the metadata is placed back into the 'description' key of the arff object,
    and thus dumped as normal description would be.
    """
    _METADATA_STRING = '@metadata'
    _METADATA_COLUMNS_COUNT = 3  # @METADATA KEY VALUE
    _METADATA_KEY_COLUMN = 1     # First key,
    _METADATA_VALUE_COLUMN = 2   # then value
    _ATTRIBUTES_TYPE = {'NUMERIC': np.float32, 'REAL': np.double, 'INTEGER': np.int64}

    def __init__(self):
        pass

    # Public interface functions (I/O)
    #
    # I. Loading functions (from file or string)
    #
    @staticmethod
    def load(fp):
        """
        Load a file-like object containing the arff document and convert it into an arff object.

        :param fp: file-like object with the arff document (ex. result of open(filename)).
        :return: arff object.

        """
        # Store this to add more lines to description
        # (the liac-arff package counts as description only the lines *before* @RELATION,
        # and we want all the lines before @DATA)
        initial_position = fp.tell()
        load_obj = arff.load(fp)

        # now the file object is again points at its start
        fp.seek(initial_position)
        load_obj['description'] = ArffHelper._extract_description(fp)

        ArffHelper._load_metadata(load_obj)
        load_obj = ArffHelper.convert_data_to_structured_array(load_obj)

        return load_obj

    @staticmethod
    def loads(s):
        """
        Convert a string instance containing the arff document into an arff object.

        :param s: string with the arff document.
        :return: arff object.

        """
        load_obj = arff.loads(s)
        # extract all of the description lines (i.e. all before @DATA, instead of the default behaviour with just
        # the lines before @RELATION being considered description)
        load_obj['description'] = ArffHelper._extract_description(s.split('\n'))
        ArffHelper._load_metadata(load_obj)
        load_obj = ArffHelper.convert_data_to_structured_array(load_obj)

        return load_obj

    # II. Dumping functions (to file or string)
    #
    @staticmethod
    def dump(obj, fp):
        """
        Serialize an object representing the arff document to a file-like object.

        :param obj: an arff object.
        :param fp: a file-like object with the arff document.

        """
        if obj['data'].size == 0:
            raise ValueError('Cannot dump an empty arff object! Use ArffHelper.add_column() to populate the object.')
        # _dump_metadata() removes the 'metadata' key, so we preserve the original state of the object here not to
        # damage the input
        dump_obj = obj.copy()
        ArffHelper._dump_metadata(dump_obj, fp)

        return arff.dump(dump_obj, fp)

    @staticmethod
    def dumps(obj):
        """
        Serialize an object representing an arff document, returning a string.

        :param obj: arff object.
        :return: string with the arff document.

        """
        if obj['data'].size == 0:
            raise ValueError('Cannot dump an empty arff object! Use ArffHelper.add_column() to populate the object.')
        # _dump_metadata() removes the 'metadata' key, so we preserve the original state of the object here not to
        # damage the input
        dump_obj = obj.copy()
        metadata_str = ArffHelper._dump_metadata(dump_obj)
        if not metadata_str:
            # if not empty, add a newline character
            metadata_str += '\n'
        return '{}{}'.format(metadata_str, arff.dumps(dump_obj))

    # Additional interface
    # III. Adding/removing columns to arff object (and its 'data' section)
    #
    @staticmethod
    def add_column(obj, name, dtype, default_value):
        """
        Add a new column to @obj['data'] and a new attribute to @obj['attributes']
        (i.e. the name of the new column and the data type for this column).
        This operation is performed in-place, so the @obj itself is changed.

        :param obj: arff object before adding new column.
        :param name: name of the new column.
        :param dtype: data type of the new column.
                      Available data types:
                      'NUMERIC', 'REAL', 'INTEGER' or a list of strings (then it's a categorical column with
                      the provided values as options).
        :param default_value: default value of the new column (we need to somehow assign the data in the new column).
        :return: arff object with an additional column.

        """
        obj['data'] = ArffHelper.add_column_to_array(obj['data'], name, dtype, default_value)
        obj['attributes'].append((name, dtype))

        return obj

    @staticmethod
    def add_column_to_array(arr, name, dtype, def_value):
        """
        Add a new column to a structured numpy array.

        :param arr: numpy array before adding column.
        :param name: name of the new column.
        :param dtype: data type of the new column.
                      Available data types:
                      'NUMERIC', 'REAL', 'INTEGER' or a list of strings (then it's a categorical column with
                      the provided values as options).
        :param def_value: default value of the new column.
        :return: numpy array with new column.

        """
        # check if def_value is in dtype
        if type(def_value) == str and def_value not in dtype:
            warnings.warn("The type of the default value is not the same as type of column data"
                          " or the default value is not in the list (date type provided is {})".format(name))

        if name in arr.dtype.names:
            raise ValueError('Array @arr already has a field {}'.format(name))

        if arr.size != 0:
            arr = rfn.append_fields(base=arr,
                                    names=name,
                                    data=[def_value] * len(arr),
                                    dtypes=ArffHelper._convert_dtype_to_numpy(dtype),
                                    usemask=False)
        else:
            # If @arr is empty, it should have been created with ArffHelper.create_empty() method, or in a similar
            # fashion. In that case, it has a length (passed as a parameter at creation), but no elements.
            arr = np.array([def_value] * len(arr), dtype=[(name, ArffHelper._convert_dtype_to_numpy(dtype))])
        return arr

    @staticmethod
    def remove_column(obj, name):
        """
        Remove a column with respective name from @obj['data'] and its attributes (@obj['attributes']).

        :param obj: arff object before adding new column.
        :param name: name of the deleted column.
        :return: arff object without the column @name.

        """
        deleted_column_index = [column_name for column_name, _ in obj['attributes']].index(name)
        obj['attributes'].pop(deleted_column_index)
        # keep just the remaining attributes
        obj['data'] = rfn.drop_fields(base=obj['data'],
                                      drop_names=name,
                                      usemask=False)
        return obj

    @staticmethod
    def rename_column(obj, old_name, new_name, new_dtype=None):
        """
        Remove a column with respective name from @obj['data'] and its attributes (@obj['attributes']).

        :param obj: arff object before adding new column.
        :param old_name: name of the renamed column.
        :param new_name: new name of the renamed column
        :param new_dtype: set the new column ARFF data type to this; if None, the data type of the original column
                          is preserved
        :return: arff object without the column @old_name changed to @new_name.

        """
        renamed_column_index = [column_name for column_name, _ in obj['attributes']].index(old_name)
        #  not to assign to a tuple, convert it to list
        dtype_names = list(obj['data'].dtype.names)
        dtype_names[renamed_column_index] = new_name
        obj['data'].dtype.names = tuple(dtype_names)
        # preserve the data type, if needed
        obj['attributes'][renamed_column_index] = (new_name,
                                                   new_dtype or obj['attributes'][renamed_column_index][1])
        return obj

    @staticmethod
    def convert_data_to_structured_array(obj):
        """
        Convert data in @obj['data'] into a structured numpy array according to the data type in
        @obj['attributes'].

        :param obj: arff object before data conversion.
        :return: arff object after data conversion.

        """
        d = np.dtype([(str(at[0]), ArffHelper._convert_dtype_to_numpy(at[1])) for at in obj['attributes']])
        obj['data'] = np.array([tuple(item) for item in obj['data']], dtype=d)
        return obj

    @staticmethod
    def create_empty(length, relation_name='gaze_labels'):
        """
        Create an empty arff object, so that more columns can be added later.

        :param length: length of the empty arff object
        :param relation_name: name of the created object relation
        :return: empty arff object

        """
        obj = {
            'relation': relation_name,
            'description': '',
            'metadata': OrderedDict(),
            'attributes': [],
            'data': np.empty(shape=(length, 0), dtype=[])
        }

        return obj

    #
    # Protected methods
    #

    @staticmethod
    def _extract_description(lines_iterable):
        """
        Extracts description (i.e. comment lines, starting with % symbol) from an iterable structure of lines
        (i.e. a file object of list of lines). All lines with a comment prefix (%) before the @DATA line are considered
        part of the description.
        The comment symbols are not considered part of the actual description text. Space characters between
        the comment sign and text are also omitted.
        :param lines_iterable: on iterable object consisting of lines of ARFF file
        :return: a description string
        """
        description_lines = []
        for line in lines_iterable:
            if line.startswith('@DATA'):
                # reached the data section, abort
                break
            if line.startswith('%'):
                # strip the comment sign and line breaks, as well as spaces in the beginning of the line
                description_lines.append(line.rstrip('\r\n')[1:].lstrip())
        return '\n'.join(description_lines)

    @staticmethod
    def _load_metadata(obj):
        """
        Looks for  '%@METADATA' keyword in the beginning of lines in @obj['description'] and extracts metadata
        names and values into a newly created @obj['metadata'] dictionary;

        :param obj: loaded arff object in form of original arff format.
        :return: arff object after extraction of %@METADATA values.

        """
        lines = obj['description'].split('\n')

        metadata = OrderedDict()
        description = []

        for i in range(len(lines)):
            if lines[i].lower().startswith(ArffHelper._METADATA_STRING):
                words = lines[i].split(' ', ArffHelper._METADATA_COLUMNS_COUNT - 1)
                if words[0].lower() != ArffHelper._METADATA_STRING:
                    warnings.warn("In line {}: potential typo in @METADATA keyword".format(i + 1))
                    continue

                if len(words) == ArffHelper._METADATA_COLUMNS_COUNT:
                    try:  # extract floating point values where possible, but not in a strict manner
                        words[ArffHelper._METADATA_VALUE_COLUMN] = float(words[ArffHelper._METADATA_VALUE_COLUMN])
                    except ValueError:
                        pass
                    metadata[words[ArffHelper._METADATA_KEY_COLUMN]] = words[ArffHelper._METADATA_VALUE_COLUMN]
                else:
                    raise ValueError("Wrong metadata format in 'description' line {}: "
                                     "should be a key-value pair separated by space".format(i + 1))
            else:
                description.append(lines[i])

        obj['metadata'] = metadata
        obj['description'] = '\n'.join(description)

        return obj

    @staticmethod
    def _dump_metadata(obj, fp=None):
        """
        Extract metadata names and values from @obj['metadata'] into either a string (if @fp is None) or the file @fp,
        and remove 'metadata' key from @obj.

        :param obj: arff object with metadata already extracted.
        :param fp: a file-like object where the arff document will be dumped, and where the metadata is written by this
                   function; if None, a string containing the metadata information.
        :return: if @fp is None, a string containing the metadata; otherwise, the @fp is returned.

        """
        if 'metadata' in obj:
            metadata_strings = []
            if len(obj['metadata']) != 0:
                for key, value in obj['metadata'].items():
                    # we dump %@METADATA strings manually to avoid space characters between '%' and '@METADATA'.
                    metadata_strings.append(' '.join(['%' + ArffHelper._METADATA_STRING,
                                                      key,
                                                      str(value)]))
            metadata_strings = '\n'.join(metadata_strings)
            del obj['metadata']
            if fp is None:
                return metadata_strings
            else:
                print(metadata_strings, file=fp)
        else:
            if fp is None:
                return ''
            else:
                return fp

    @staticmethod
    def _convert_dtype_to_numpy(data_type):
        """
        Validate input @data_type as ARFF-supported data type and convert to numpy.dtype.

        :param data_type: input data_type, string.
                          Available data types:
                          'NUMERIC', 'REAL', 'INTEGER' or a list of string (then it's a categorical attribute).
        :return: converted numpy.dtype from input data_type.

        """
        if data_type in list(ArffHelper._ATTRIBUTES_TYPE.keys()):
            return ArffHelper._ATTRIBUTES_TYPE[data_type]
        else:
            if type(data_type) == list:
                max_length = max(list(map(len, data_type)))
            else:
                raise ValueError("Wrong data type in attributes. "
                                 "It should be a list of strings or one of the data types in {}".format(
                                  ', '.join(list(ArffHelper._ATTRIBUTES_TYPE.keys()))))
            return '|U{}'.format(max_length)
