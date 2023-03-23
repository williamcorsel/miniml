import fiftyone as fo
import fiftyone.utils.data as foud


class PytorchClassificationParser(foud.LabeledImageSampleParser):
    """Parser for image classification samples loaded from a PyTorch dataset.

    This parser can parse samples from a ``torch.utils.data.DataLoader`` that
    emits ``(img_tensor, target)`` tuples, where::

        - `img_tensor`: is a PyTorch Tensor containing the image
        - `target`: the integer index of the target class

    Args:
        classes: the list of class label strings
    """

    def __init__(self, classes):
        self.classes = classes

    @property
    def has_image_path(self):
        """Whether this parser produces paths to images on disk for samples
        that it parses.
        """
        return False

    @property
    def has_image_metadata(self):
        """Whether this parser produces
        :class:`fiftyone.core.metadata.ImageMetadata` instances for samples
        that it parses.
        """
        return False

    @property
    def label_cls(self):
        """The :class:`fiftyone.core.labels.Label` class(es) returned by this
        parser.

        This can be any of the following:

        -   a :class:`fiftyone.core.labels.Label` class. In this case, the
            parser is guaranteed to return labels of this type
        -   a list or tuple of :class:`fiftyone.core.labels.Label` classes. In
            this case, the parser can produce a single label field of any of
            these types
        -   a dict mapping keys to :class:`fiftyone.core.labels.Label` classes.
            In this case, the parser will return label dictionaries with keys
            and value-types specified by this dictionary. Not all keys need be
            present in the imported labels
        -   ``None``. In this case, the parser makes no guarantees about the
            labels that it may return
        """
        return fo.Classification

    def get_image(self):
        """Returns the image from the current sample.

        Returns:
            a numpy image
        """
        img_tensor = self.current_sample[0]
        return img_tensor.cpu().numpy()

    def get_label(self):
        """Returns the label for the current sample.

        Returns:
            a :class:`fiftyone.core.labels.Label` instance, or a dictionary
            mapping field names to :class:`fiftyone.core.labels.Label`
            instances, or ``None`` if the sample is unlabeled
        """
        target = self.current_sample[1]
        return fo.Classification(label=self.classes[int(target)])