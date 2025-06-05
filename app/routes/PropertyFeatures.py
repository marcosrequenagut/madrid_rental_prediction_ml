from pydantic import BaseModel

class PropertyFeatures(BaseModel):
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