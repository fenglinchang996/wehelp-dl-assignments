from urllib import request
from html.parser import HTMLParser
from enum import Enum
from typing import TypedDict

BASE_URL = "https://24h.pchome.com.tw"


class ProductElement(Enum):
    PRODUCT_ID = "product_id"
    PRODUCT_TITLE = "product_title"
    PRODUCT_PRICE = "product_price"
    PRODUCT_REVIEW_COUNT = "product_review_count"


class Product(TypedDict):
    id: str
    title: str | None
    price: int | None
    review_count: int | None
    url: str | None


class PageParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.page_url_list = []

    @staticmethod
    def get_page_url(tag: str, attrs: list[tuple[str, str | None]]):
        if tag == "a":
            page_info = {"is_page": False, "page_url": ""}
            for key, val in attrs:
                if key == "class" and val is not None and "c-pagination__link" in val:
                    page_info["is_page"] = True
                elif key == "href" and val is not None:
                    page_info["page_url"] = val
            if page_info["is_page"]:
                return page_info["page_url"]

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        page_url = self.get_page_url(tag, attrs)
        if page_url:
            self.page_url_list.append(page_url)


class ProductListParser(HTMLParser):
    _VOID_ELEMENTS = {
        "img",
        "br",
        "input",
        "meta",
        "link",
        "hr",
        "source",
        "embed",
        "area",
        "base",
        "col",
        "track",
        "wbr",
    }

    def __init__(self):
        super().__init__()
        self.products: dict[str, Product] = {}
        self.current_product_id = ""
        self._tag_stack = []

    @staticmethod
    def get_product_element_type(
        tag: str, attrs: list[tuple[str, str | None]]
    ) -> ProductElement | None:
        if tag == "a" and any(
            key == "class" and val is not None and "c-prodInfoV2__link" in val
            for key, val in attrs
        ):
            return ProductElement.PRODUCT_ID
        elif (
            tag == "h3"
            and any(
                key == "class" and val is not None and "c-prodInfoV2__title" in val
                for key, val in attrs
            )
            and any(
                key == "data-regression" and val == "store_prodName"
                for key, val in attrs
            )
        ):
            return ProductElement.PRODUCT_TITLE
        elif tag == "div" and any(
            key == "class"
            and val is not None
            and "c-prodInfoV2__priceValue c-prodInfoV2__priceValue--m" in val
            for key, val in attrs
        ):
            return ProductElement.PRODUCT_PRICE
        elif tag == "div" and any(
            key == "class"
            and val is not None
            and "c-prodInfoV2__text c-prodInfoV2__text--xs500GrayDark" in val
            for key, val in attrs
        ):
            return ProductElement.PRODUCT_REVIEW_COUNT
        else:
            return None

    @staticmethod
    def get_product_url(tag: str, attrs: list[tuple[str, str | None]]) -> str | None:
        elementType = ProductListParser.get_product_element_type(tag, attrs)
        if elementType == ProductElement.PRODUCT_ID:
            for key, val in attrs:
                if key == "href" and val is not None:
                    return val

    @staticmethod
    def get_product_id(product_url: str | None) -> str | None:
        if product_url is not None:
            # url would be like /prod/PRODUCT_ID?fq=/S/DSAA31
            product_id = product_url.partition("?")[0].split("/")[-1]
            return product_id

    @staticmethod
    def parse_price(priceString: str) -> int:
        return int(priceString.replace("$", "").replace(",", ""))

    @staticmethod
    def parse_review_count(review_count_string: str) -> int:
        return int(review_count_string.replace("(", "").replace(")", ""))

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in self._VOID_ELEMENTS:
            return

        elementType = self.get_product_element_type(tag, attrs)
        if elementType is None:
            return

        if elementType == ProductElement.PRODUCT_ID:
            product_url = self.get_product_url(tag, attrs)
            product_id = self.get_product_id(product_url)
            if product_id:
                self.current_product_id = product_id
                self.products[product_id] = {
                    "id": product_id,
                    "title": None,
                    "price": None,
                    "review_count": None,
                    "url": product_url,
                }

        self._tag_stack.append({"tag": tag, "type": elementType})

    def handle_data(self, data: str) -> None:
        if self._tag_stack:
            if self._tag_stack[-1]["type"] == ProductElement.PRODUCT_PRICE:
                self.products[self.current_product_id]["price"] = self.parse_price(data)
            elif self._tag_stack[-1]["type"] == ProductElement.PRODUCT_TITLE:
                self.products[self.current_product_id]["title"] = data
            elif self._tag_stack[-1]["type"] == ProductElement.PRODUCT_REVIEW_COUNT:
                self.products[self.current_product_id]["review_count"] = (
                    self.parse_review_count(data)
                )

    def handle_endtag(self, tag: str) -> None:
        if self._tag_stack and self._tag_stack[-1]["tag"] == tag:
            self._tag_stack.pop(-1)


class ProductRatingParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self._is_product_rating_tag = False
        self.product_rating = 0

    @staticmethod
    def check_product_rating_tag(tag: str, attrs: list[tuple[str, str | None]]):
        if tag == "div":
            for key, val in attrs:
                if (
                    key == "class"
                    and val is not None
                    and "c-ratingIcon__textNumber" in val
                ):
                    return True
        return False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if self.check_product_rating_tag(tag, attrs):
            self._is_product_rating_tag = True

    def handle_data(self, data: str) -> None:
        if self._is_product_rating_tag:
            self.product_rating = float(data)
            self._is_product_rating_tag = False


def get_html_content(URL: str) -> str | None:
    req = request.Request(URL)
    try:
        with request.urlopen(req) as f:
            content_charset = f.headers.get_content_charset()
            charset = content_charset if content_charset else "utf-8"
            html_content = f.read().decode(charset)
        return html_content
    except Exception as e:
        print(f"Error: {e}")
        return None


# Create products.txt and write product IDs into it
def list_product_ids(product_list: list[Product]):
    with open("products.txt", mode="w") as f:
        for product in product_list:
            f.write(f"{product['id']}\n")


# Find i5 processor products and calculate their average price
def print_i5_average_price(product_list: list[Product]):
    i5_products_price = [
        product["price"]
        for product in product_list
        if product["title"]
        and "i5" in product["title"]
        and product["price"] is not None
    ]
    i5_products_average_price = (
        sum(i5_products_price) / len(i5_products_price) if i5_products_price else 0
    )
    print(i5_products_average_price)


# Find best products (at least 1 review and rating greater than 4.9)
def find_best_product_ids(product_list: list[Product]):
    products_with_review = [
        product
        for product in product_list
        if product["review_count"] and product["review_count"] > 0
    ]
    best_product_list = []
    # It seems no rating info in product list page,
    # having to fetch each product page to get precise rating
    for product in products_with_review:
        if product["url"] is not None:
            product_html_content = get_html_content(BASE_URL + product["url"])
            if product_html_content:
                product_rating_parser = ProductRatingParser()
                product_rating_parser.feed(product_html_content)
                if product_rating_parser.product_rating > 4.9:
                    best_product_list.append(product)
    with open("best-products.txt", mode="w") as f:
        for product in best_product_list:
            f.write(f"{product['id']}\n")


def calculate_z_score(product_list: list[Product]):
    # Calculate z-score and write them to standardization.csv
    prices = [
        product["price"] for product in product_list if product["price"] is not None
    ]
    if not prices:
        return
    mu = sum(prices) / len(prices)
    sigma = (sum([(price - mu) ** 2 for price in prices]) / len(prices)) ** 0.5
    with open("standardization.csv", mode="w") as f:
        for product in product_list:
            if product["price"] is not None:
                z_score = (product["price"] - mu) / sigma
                f.write(f"{product['id']},{product['price']},{z_score}\n")


def main():
    INIT_URL = BASE_URL + "/store/DSAA31"
    init_content = get_html_content(INIT_URL)
    if not init_content:
        return

    pageParser = PageParser()
    pageParser.feed(init_content)
    if not pageParser.page_url_list:
        return

    product_list: list[Product] = []

    for page_url in pageParser.page_url_list:
        page_content = get_html_content(BASE_URL + page_url)
        if not page_content:
            return

        product_list_parser = ProductListParser()
        product_list_parser.feed(page_content)
        if not product_list_parser.products:
            return
        for val in product_list_parser.products.values():
            product_list.append(val)

    list_product_ids(product_list)

    find_best_product_ids(product_list)

    print_i5_average_price(product_list)

    calculate_z_score(product_list)


if __name__ == "__main__":
    main()
