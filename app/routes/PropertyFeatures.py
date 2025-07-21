from pydantic import BaseModel

class PropertyFeatures(BaseModel):

    """
        Define the required features of a property to be used for price prediction.

        This model is used by the prediction endpoint in `predict.py`. The user must
        manually provide all these fields when calling the API. These fields represent
        the characteristics of a property, which will be passed to a trained machine
        learning model to estimate the property's price.

        ## Attributes (Input Features)
        :param constructed_area: (float) Total constructed area of the property in square meters.
        :param has_terrace: (int) 1 if the property has a terrace, 0 otherwise.
        :param is_parkingspace_included: (int) 1 if a parking space is included in the price, 0 otherwise.
        :param number_of_rooms: (int) Total number of rooms in the property.
        :param number_of_bathrooms: (int) Total number of bathrooms in the property.
        :param has_swimming_pool: (int) 1 if the property has a swimming pool, 0 otherwise.
        :param is_top_floor: (int) 1 if the property is located on the top floor, 0 otherwise.
        :param distance_to_city_center: (float) Distance from the property to the city center (in kilometers or meters, depending on the model).
        :param distance_to_city_metro: (float) Distance from the property to the nearest metro station.
        :param distance_to_city_castellana: (float) Distance from the property to Paseo de la Castellana.
        :param constructed_year: (int) The year the property was constructed.
        :param floorclean: (int) Numerical representation of the floor the property is on.
        :param location: (str) Neighbourhood identifier or name (used as a categorical feature).
        :param district: (str) Name of the district where the property is located.

        ## Usage
        This schema is used to validate and structure the input data sent to the API
        before passing it to the machine learning model for price prediction.
    """
    constructed_area: float
    has_terrace: int
    is_parkingspace_included: int
    number_of_rooms: int
    number_of_bathrooms: int
    has_swimming_pool: int
    is_top_floor: int
    distance_to_city_center: float
    distance_to_city_metro: float
    distance_to_city_castellana: float
    constructed_year: int
    floorclean: int
    location: str
    district: str