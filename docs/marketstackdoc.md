API Documentation
Note: All the new functionalities and new endpoints listed in the pricing plans are designed for use with the new API endpoints in Version V2. Please refer to the V2 API endpoints documentation page for proper usage of the service. Please be aware that Version V1 of the API endpoints will be deprecated for use after June 30th, 2025

Welcome to the marketstack API documentation! In the following series of articles you will learn how to query the marketstack JSON API for real-time, intraday and historical stock market data, define multiple stock symbols, retrieve extensive data about 70+ stock exchanges, 170.000+ stock tickers from more than 50 countries, as well as 750+ market indices, information about timezones, currencies, and more.

Our API is built upon a RESTful and easy-to-understand request and response structure. API requests are always sent using a simple API request URL with a series of required and optional HTTP GET parameters, and API responses are provided in lightweight JSON format. Continue below to get started, or click the blue button above to jump to our 3-Step Quickstart Guide.

Run in postman
Fork collection into your workspace
Getting Started
API Authentication
For every API request you make, you will need to make sure to be authenticated with the API by passing your API access key to the API's access_key parameter. You can find an example below.

Example API Request:

Run API Requesthttp://api.marketstack.com/v1/eod
    ? access_key = YOUR_MARKETSTACK_API_KEY
    & symbols = AAPL

Important: Please make sure not to expose your API access key publicly. If you believe your API access key may be compromised, you can always reset in your account dashboard.

256-bit HTTPS EncryptionAvailable on: All Plans
If you're subscribed to either the free or any paid plans, you will be able to access the marketstack API using industry-standard HTTPS. To do that, simply use the https protocol when making API requests.

Example API Request:

https://api.marketstack.com/v1

API Errors
API errors consist of error code and message response objects. If an error occurs, the marketstack will return HTTP status codes, such as 404 for "not found" errors. If your API request succeeds, status code 200 will be sent.

For validation errors, the marketstack API will also provide a context response object returning additional information about the error that occurred in the form of one or multiple sub-objects, each equipped with the name of the affected parameter as well as key and message objects. You can find an example error below.

Example Error:

{
   "error": {
      "code": "validation_error",
      "message": "You have to specify at least one symbol and not more than 100",
   }
}

Common API Errors:

Code	Type	Description
401	Unauthorized	Check your access key or activity of the account
403	function_access_restricted	The given API endpoint is not supported on the current subscription plan.
404	invalid_api_function	The given API endpoint does not exist.
404	404_not_found	Resource not found.
429	too_many_requests	The given user account has reached its monthly allowed request volume.
429	rate_limit_reached	The given user account has reached the rate limit.
500	internal_error	An internal error occurred.

Note: The api is limited to 5 requests per second.

Supported Endpoint
End-of-Day DataAvailable on: All plans
You can use the API's eod endpoint in order to obtain end-of-day data for one or multiple stock tickers. A single or multiple comma-separated ticker symbols are passed to the API using the symbols parameter.

Note: To request end-of-day data for single ticker symbols, you can also use the API's Tickers Endpoint.

Example API Request:

Run API Requesthttp://api.marketstack.com/v1/eod
    ? access_key = YOUR_MARKETSTACK_API_KEY
    & symbols = AAPL

Endpoint Features:

Object	Description
/eod/[date]	Specify a date in YYYY-MM-DD format. You can also specify an exact time in ISO-8601 date format, e.g. 2020-05-21T00:00:00+0000. Example: /eod/2020-01-01
/eod/latest	Obtain the latest available end-of-day data for one or multiple stock tickers.

Parameters:

Object	Description
access_key	[Required] Specify your API access key, available in your account dashboard.
symbols	[Required] Specify one or multiple comma-separated stock symbols (tickers) for your request, e.g. AAPL or AAPL,MSFT. Each symbol consumes one API request. Maximum: 100 symbols
exchange	[Optional] Filter your results based on a specific stock exchange by specifying the MIC identification of a stock exchange. Example: XNAS
sort	[Optional] By default, results are sorted by date/time descending. Use this parameter to specify a sorting order. Available values: DESC (Default), ASC.
date_from	[Optional] Filter results based on a specific timeframe by passing a from-date in YYYY-MM-DD format. You can also specify an exact time in ISO-8601 date format, e.g. 2020-05-21T00:00:00+0000.
date_to	[Optional] Filter results based on a specific timeframe by passing an end-date in YYYY-MM-DD format. You can also specify an exact time in ISO-8601 date format, e.g. 2020-05-21T00:00:00+0000.
limit	[Optional] Specify a pagination limit (number of results per page) for your API request. Default limit value is 100, maximum allowed limit value is 1000.
offset	[Optional] Specify a pagination offset value for your API request. Example: An offset value of 100 combined with a limit value of 10 would show results 100-110. Default value is 0, starting with the first available result.

Example API Response:

If your API request was successful, the marketstack API will return both pagination information as well as a data object, which contains a separate sub-object for each requested date/time and symbol. All response objects are explained below.

{
    "pagination": {
        "limit": 100,
        "offset": 0,
        "count": 100,
        "total": 9944
    },
    "data": [
        {
            "open": 129.8,
            "high": 133.04,
            "low": 129.47,
            "close": 132.995,
            "volume": 106686703.0,
            "adj_high": 133.04,
            "adj_low": 129.47,
            "adj_close": 132.995,
            "adj_open": 129.8,
            "adj_volume": 106686703.0,
            "split_factor": 1.0,
            "dividend": 0.0,
            "symbol": "AAPL",
            "exchange": "XNAS",
            "date": "2021-04-09T00:00:00+0000"
            },
            [...]
    ]
}

API Response Objects:

Response Object	Description
pagination > limit	Returns your pagination limit value.
pagination > offset	Returns your pagination offset value.
pagination > count	Returns the results count on the current page.
pagination > total	Returns the total count of results available.
date	Returns the exact UTC date/time the given data was collected in ISO-8601 format.
symbol	Returns the stock ticker symbol of the current data object.
exchange	Returns the exchange MIC identification associated with the current data object.
split_factor	Returns the split factor, which is used to adjust prices when a company splits, reverse splits, or pays a distribution.
dividend	Returns the dividend, which are the distribution of earnings to shareholders.
open	Returns the raw opening price of the given stock ticker.
high	Returns the raw high price of the given stock ticker.
low	Returns the raw low price of the given stock ticker.
close	Returns the raw closing price of the given stock ticker.
volume	Returns the raw volume of the given stock ticker.
adj_open	Returns the adjusted opening price of the given stock ticker.
adj_high	Returns the adjusted high price of the given stock ticker.
adj_low	Returns the adjusted low price of the given stock ticker.
adj_close	Returns the adjusted closing price of the given stock ticker.
adj_volume	Returns the adjusted volume of given stock ticker.

Adjusted Prices: "Adjusted" prices are stock price values that were amended to accurately reflect the given stock's value after accounting for any corporate actions, such as splits or dividends. Adjustments are made in accordance with the "CRSP Calculations" methodology set forth by the Center for Research in Security Prices (CRSP).


End-of-Day Data:

You can use the API's eod endpoint in order to obtain end-of-day data for one or multiple stock tickers. A single or multiple comma-separated ticker symbols are passed to the API using the symbols parameter.

JavaScript FetchJavaScript AxiosPython RequestsPython Http.client
                

const url = "https://api.marketstack.com/v1/eod?access_key={PASTE_YOUR_API_KEY_HERE}&symbols=AAPL";
const options = {
    method: "GET",
};

try {
    const response = await fetch(url, options);
    const result = await response.text();
    console.log(result);
} catch (error) {
    console.error(error);
}


            


Market IndicesAvailable on: Basic Plan and higher
The API is also capable of delivering end-of-day for 750+ of the world's major indices, including the S&P 500 Index, the Dow Jones Industrial Average Index as well as the NASDAQ Composite Index. Index data is available both on a "latest" basis as well as historically.

To list or access index data, simply pass INDX as your stock exchange MIC identification, as seen in the examples below. The example API request below illustrates how to obtain end-of-day data for the DJI market index.

Example API Request:

Run API Requesthttp://api.marketstack.com/v1/eod
    ? access_key = YOUR_MARKETSTACK_API_KEY
    & symbols = DJI.INDX

Parameters:

For more information about request parameters, please refer to the End-of-day Data section of this documentation.


API Response:

{
    "pagination": {
        "limit": 1,
        "offset": 0,
        "count": 1,
        "total": 7561
    },
    "data": [
        {
            "date": "2020-08-21T00:00:00+0000",
            "symbol": "DJI.INDX",
            "exchange": "INDX",
            "open": 27758.1309,
            "high": 27959.4805,
            "low": 27686.7793,
            "close": 27930.3301,
            "volume": 374339179,
            "adj_high": null,
            "adj_low": null,
            "adj_close": 27930.3301,
            "adj_open": null,
            "adj_volume": null
        }
    ]
}

API Response Objects:

For more information about response objects, please refer to the End-of-day Data section of this documentation.


Market Indices in Other API Endpoints:

Object	Description
/exchanges/INDX/tickers	Obtain all available market indices by passing INDX as the exchange MIC identification.
/tickers/[symbol].INDX	Obtain meta information for a specific market index.
/tickers/[symbol].INDX/eod	Obtain end-of-day data for a specific market index.

Historical DataAvailable on: All plans
Historical stock prices are available both from the end-of-day (eod) and intraday (intraday) API endpoints. To obtain historical data, simply use the date_from and date_to parameters as shown in the example request below.

Example API Request:

Run API Requesthttp://api.marketstack.com/v1/eod
    ? access_key = YOUR_MARKETSTACK_API_KEY
    & symbols = AAPL
    & date_from = 2025-07-25
    & date_to = 2025-08-04

Parameters:

For details on request parameters on the eod data endpoint, please jump to the End-of-Day Data section.

Example API Response:

{
    "pagination": {
        "limit": 100,
        "offset": 0,
        "count": 22,
        "total": 22
    },
    "data": [
        {
            "date": "2019-02-01T00:00:00+0000",
            "symbol": "AAPL",
            "exchange": "XNAS",
            "open": 166.96,
            "high": 168.98,
            "low": 165.93,
            "close": 166.52,
            "volume": 32668138.0,
            "adj_open": 164.0861621594,
            "adj_high": 166.0713924395,
            "adj_low": 163.073891274,
            "adj_close": 163.6537357617,
            "adj_volume": 32668138.0
        },
        {
            "date": "2019-01-31T00:00:00+0000",
            "symbol": "AAPL",
            "exchange": "XNAS",
            "open": 166.11,
            "high": 169.0,
            "low": 164.56,
            "close": 166.44,
            "volume": 40739649.0,
            "adj_open": 163.2507929821,
            "adj_high": 166.0910481848,
            "adj_low": 161.7274727177,
            "adj_close": 163.5751127804,
            "adj_volume": 40739649.0
        }
        [...]
    ]
}

API Response Objects:

For details on API response objects, please jump to the End-of-Day Data section.

Note: Historical end-of-day data (eod) is available for up to 30 years back, while intraday data (intraday) always only offers the last 10,000 entries for each of the intervals available. Example: For a 1-minute interval, historical intraday data is available for up to 10,000 minutes back.


Historial Data:

Historical stock prices are available both from the end-of-day (eod) and intraday (intraday) API endpoints. To obtain historical data, simply use the date_from and date_to parameters as shown in the example request below.

JavaScript FetchJavaScript AxiosPython RequestsPython Http.client
                

const url =
    "https://api.marketstack.com/v1/eod?access_key={PASTE_YOUR_API_KEY_HERE}&symbols=AAPL&date_from=2024-02-26&date_to=2024-03-07";
const options = {
    method: "GET",
};

try {
    const response = await fetch(url, options);
    const result = await response.text();
    console.log(result);
} catch (error) {
    console.error(error);
}

                
            


Intraday DataAvailable on: Basic Plan and higher
In additional to daily end-of-day stock prices, the marketstack API also supports intraday data with data intervals as short as one minute. Intraday prices are available for all US stock tickers included in the IEX (Investors Exchange) stock exchange.

To obtain intraday data, you can use the API's intraday endpoint and specify your preferred stock ticker symbols.

Note: To request intraday data for single ticker symbols, you can also use the API's Tickers Endpoint.

Example API Request:

Run API Requesthttp://api.marketstack.com/v1/intraday
    ? access_key = YOUR_MARKETSTACK_API_KEY
    & symbols = AAPL

Endpoint Features:

Object	Description
/intraday/[date]	Specify a date in YYYY-MM-DD format. You can also specify an exact time in ISO-8601 date format, e.g. 2020-05-21T00:00:00+0000. Example: /intraday/2020-01-01
/intraday/latest	Obtain the latest available intraday data for one or multiple stock tickers.

Parameters:

Object	Description
access_key	[Required] Specify your API access key, available in your account dashboard.
symbols	[Required] Specify one or multiple comma-separated stock symbols (tickers) for your request, e.g. AAPL or AAPL,MSFT. Each symbol consumes one API request. Maximum: 100 symbols
exchange	[Optional] Filter your results based on a specific stock exchange by specifying the MIC identification of a stock exchange. Example: IEXG
interval	[Optional] Specify your preferred data interval. Available values: 1min, 5min, 10min, 15min, 30min, 1hour (Default), 3hour, 6hour, 12hour and 24hour.
sort	[Optional] By default, results are sorted by date/time descending. Use this parameter to specify a sorting order. Available values: DESC (Default), ASC.
date_from	[Optional] Filter results based on a specific timeframe by passing a from-date in YYYY-MM-DD format. You can also specify an exact time in ISO-8601 date format, e.g. 2020-05-21T00:00:00+0000.
date_to	[Optional] Filter results based on a specific timeframe by passing an end-date in YYYY-MM-DD format. You can also specify an exact time in ISO-8601 date format, e.g. 2020-05-21T00:00:00+0000.
limit	[Optional] Specify a pagination limit (number of results per page) for your API request. Default limit value is 100, maximum allowed limit value is 1000.
offset	[Optional] Specify a pagination offset value for your API request. Example: An offset value of 100 combined with a limit value of 10 would show results 100-110. Default value is 0, starting with the first available result.

Real-Time Updates: Please note that data frequency intervals below 15 minutes (15min) are only supported if you are subscribed to the Professional Plan or higher. If you are the Free or Basic Plan, please upgrade your account.

Example API Response:

If your API request was successful, the marketstack API will return both pagination information as well as a data object, which contains a separate sub-object for each requested date/time and symbol. All response objects are explained below.

{
    "pagination": {
        "limit": 100,
        "offset": 0,
        "count": 100,
        "total": 5000
    },
    "data": [
        {
            "date": "2020-06-02T00:00:00+0000"
            "symbol": "AAPL",
            "exchange": "IEXG",
            "open": 317.75,
            "high": 322.35,
            "low": 317.21,
            "close": 317.94,
            "last": 318.91,
            "volume": 41551000
        },
        [...]
    ]
}

API Response Objects:

Response Object	Description
pagination > limit	Returns your pagination limit value.
pagination > offset	Returns your pagination offset value.
pagination > count	Returns the results count on the current page.
pagination > total	Returns the total count of results available.
date	Returns the exact UTC date/time the given data was collected in ISO-8601 format.
symbol	Returns the stock ticker symbol of the current data object.
exchange	Returns the exchange MIC identification associated with the current data object.
open	Returns the raw opening price of the given stock ticker.
high	Returns the raw high price of the given stock ticker.
low	Returns the raw low price of the given stock ticker.
close	Returns the raw closing price of the given stock ticker.
last	Returns the last executed trade of the given symbol on its exchange.
volume	Returns the volume of the given stock ticker.

Intraday Data:

JavaScript FetchJavaScript AxiosPython RequestsPython Http.client
                

const url = "https://api.marketstack.com/v1/intraday?access_key={PASTE_YOUR_API_KEY_HERE}&symbols=AAPL";
const options = {
    method: "GET",
};

try {
    const response = await fetch(url, options);
    const result = await response.text();
    console.log(result);
} catch (error) {
    console.error(error);
}

                
            


Real-Time Updates

For customers with an active subscription to the Professional Plan, the marketstack API's intraday endpoint is also capable of providing real-time market data, updated every minute, every 5 minutes or every 10 minutes.

To obtain real-time data using this endpoint, simply append the API's interval parameter and set it to 1min, 5min or 10min.

Example API Request:

Run API Requesthttp://api.marketstack.com/v1/intraday
    ? access_key = YOUR_MARKETSTACK_API_KEY
    & symbols = AAPL
    & interval = 1min

Endpoint Features, Parameters & API Response:

To learn about endpoint features, request parameters and API response objects, please navigate to the Intraday Data section.


Real-Time-Updates:

Specify the interval parameter and set it to 1min, 5min or 10min to obtain real-time data using this endpoint.

JavaScript FetchJavaScript AxiosPython RequestsPython Http.client
                

const url = "https://api.marketstack.com/v1/intraday?access_key={PASTE_YOUR_API_KEY_HERE}&symbols=AAPL&interval=1min";
const options = {
    method: "GET",
};

try {
    const response = await fetch(url, options);
    const result = await response.text();
    console.log(result);
} catch (error) {
    console.error(error);
}

                
            


Splits DataAvailable on: All plans
Using the APIssplitsendpoint you will be able to look up information about the stock splits factor for different symbols. You will be able to find and try out an example API request below.

To obtain splits data, you can use the API's splits endpoint and specify your preferred stock ticker symbols.

Note: To request splits data for single ticker symbols, you can also use the API's Tickers Endpoint.

Example API Request:

Run API Requesthttp://api.marketstack.com/v1/splits
    ? access_key = YOUR_MARKETSTACK_API_KEY
    & symbols = AAPL

Endpoint Features:


Parameters:

Object	Description
access_key	[Required] Specify your API access key, available in your account dashboard.
symbols	[Required] Specify one or multiple comma-separated stock symbols (tickers) for your request, e.g. AAPL or AAPL,MSFT. Each symbol consumes one API request. Maximum: 100 symbols
sort	[Optional] By default, results are sorted by date/time descending. Use this parameter to specify a sorting order. Available values: DESC (Default), ASC.
date_from	[Optional] Filter results based on a specific timeframe by passing a from-date in YYYY-MM-DD format. You can also specify an exact time in ISO-8601 date format, e.g. 2020-05-21T00:00:00+0000.
date_to	[Optional] Filter results based on a specific timeframe by passing an end-date in YYYY-MM-DD format. You can also specify an exact time in ISO-8601 date format, e.g. 2020-05-21T00:00:00+0000.
limit	[Optional] Specify a pagination limit (number of results per page) for your API request. Default limit value is 100, maximum allowed limit value is 1000.
offset	[Optional] Specify a pagination offset value for your API request. Example: An offset value of 100 combined with a limit value of 10 would show results 100-110. Default value is 0, starting with the first available result.

Example API Response:

If your API request was successful, the marketstack API will return both pagination information as well as a data object, which contains a separate sub-object for each requested date/time and symbol. All response objects are explained below.

{
    "pagination": {
        "limit": 100,
        "offset": 0,
        "count": 100,
        "total": 50765
    },
    "data": [
        {
            "date": "2021-05-24",
            "split_factor": 0.5,
            "symbol": "IAU"
        },
        [...]
    ]
}

API Response Objects:

Response Object	Description
pagination > limit	Returns your pagination limit value.
pagination > offset	Returns your pagination offset value.
pagination > count	Returns the results count on the current page.
pagination > total	Returns the total count of results available.
date	Returns the exact UTC date/time the given data was collected in ISO-8601 format.
symbol	Returns the stock ticker symbol of the current data object.
volume	Returns the split factor for that symbol on the date.

Splits Data:

JavaScript FetchJavaScript AxiosPython RequestsPython Http.client
                

const url = 'https://api.marketstack.com/v1/splits?access_key={PASTE_YOUR_API_KEY_HERE}&symbols=AAPL';
const options = {
	method: 'GET'
};

try {
	const response = await fetch(url, options);
	const result = await response.text();
	console.log(result);
} catch (error) {
	console.error(error);
}

                
            


Dividends DataAvailable on: All plans
Using the APIsdividendsendpoint you will be able to look up information about the stock dividend for different symbols. You will be able to find and try out an example API request below.

To obtain dividends data, you can use the API's dividends endpoint and specify your preferred stock ticker symbols.

Note: To request dividends data for single ticker symbols, you can also use the API's Tickers Endpoint.

Example API Request:

Run API Requesthttp://api.marketstack.com/v1/dividends
    ? access_key = YOUR_MARKETSTACK_API_KEY
    & symbols = AAPL

Endpoint Features:


Parameters:

Object	Description
access_key	[Required] Specify your API access key, available in your account dashboard.
symbols	[Required] Specify one or multiple comma-separated stock symbols (tickers) for your request, e.g. AAPL or AAPL,MSFT. Each symbol consumes one API request. Maximum: 100 symbols
sort	[Optional] By default, results are sorted by date/time descending. Use this parameter to specify a sorting order. Available values: DESC (Default), ASC.
date_from	[Optional] Filter results based on a specific timeframe by passing a from-date in YYYY-MM-DD format. You can also specify an exact time in ISO-8601 date format, e.g. 2020-05-21T00:00:00+0000.
date_to	[Optional] Filter results based on a specific timeframe by passing an end-date in YYYY-MM-DD format. You can also specify an exact time in ISO-8601 date format, e.g. 2020-05-21T00:00:00+0000.
limit	[Optional] Specify a pagination limit (number of results per page) for your API request. Default limit value is 100, maximum allowed limit value is 1000.
offset	[Optional] Specify a pagination offset value for your API request. Example: An offset value of 100 combined with a limit value of 10 would show results 100-110. Default value is 0, starting with the first available result.

Example API Response:

If your API request was successful, the marketstack API will return both pagination information as well as a data object, which contains a separate sub-object for each requested date/time and symbol. All response objects are explained below.

{
    "pagination": {
        "limit": 100,
        "offset": 0,
        "count": 100,
        "total": 50765
    },
    "data": [
        {
            "date": "2021-05-24",
            "dividend": 0.5,
            "symbol": "IAU"
        },
        [...]
    ]
}

API Response Objects:

Response Object	Description
pagination > limit	Returns your pagination limit value.
pagination > offset	Returns your pagination offset value.
pagination > count	Returns the results count on the current page.
pagination > total	Returns the total count of results available.
date	Returns the exact UTC date/time the given data was collected in ISO-8601 format.
symbol	Returns the stock ticker symbol of the current data object.
volume	Returns the dividend for that symbol on the date.

Dividends Data:

JavaScript FetchJavaScript AxiosPython RequestsPython Http.client
                

const url = 'https://api.marketstack.com/v1/dividends?access_key={PASTE_YOUR_API_KEY_HERE}&symbols=AAPL';
const options = {
	method: 'GET'
};

try {
	const response = await fetch(url, options);
	const result = await response.text();
	console.log(result);
} catch (error) {
	console.error(error);
}

                
            


TickersAvailable on: All plans
Using the API's tickers endpoint you will be able to look up information about one or multiple stock ticker symbols as well as obtain end-of-day, real-time and intraday market data for single tickers. You will be able to find and try out an example API request below.

Example API Request:

Run API Requesthttp://api.marketstack.com/v1/tickers
    ? access_key = YOUR_MARKETSTACK_API_KEY

Endpoint Features:

Object	Description
/tickers/[symbol]	Obtain information about a specific ticker symbol by attach it to your API request URL, e.g. /tickers/AAPL.
/tickers/[symbol]/eod	Obtain end-of-day data for a specific stock ticker by attaching /eod to your URL, e.g. /tickers/AAPL/eod. This route supports parameters of the End-of-day Data endpoint.
/tickers/[symbol]/splits	Obtain end-of-day data for a specific stock ticker by attaching /splits to your URL, e.g. /tickers/AAPL/splits. This route supports parameters like date period date_from and date_to and also you can sort the results DESC or ASC.
/tickers/[symbol]/dividends	Obtain end-of-day data for a specific stock ticker by attaching /dividends to your URL, e.g. /tickers/AAPL/dividends. This route supports parameters like date period date_from and date_to and also you can sort the results DESC or ASC.
/tickers/[symbol]/intraday	Obtain real-time & intraday data for a specific stock ticker by attaching /intraday to your URL, e.g. /tickers/AAPL/intraday. This route supports parameters of the Intraday Data endpoint.
/tickers/[symbol]/eod/[date]	Specify a date in YYYY-MM-DD format. You can also specify an exact time in ISO-8601 date format, e.g. 2020-05-21T00:00:00+0000. Example: /eod/2020-01-01 or /intraday/2020-01-01
/tickers/[symbol]/eod/latest	Obtain the latest end-of-day data for a given stock symbol. Example: /tickers/AAPL/eod/latest
/tickers/[symbol]/intraday/latest	Obtain the latest intraday data for a given stock symbol. Example: /tickers/AAPL/intraday/latest

Parameters:

Object	Description
access_key	[Required] Specify your API access key, available in your account dashboard.
exchange	[Optional] To filter your results based on a specific stock exchange, use this parameter to specify the MIC identification of a stock exchange. Example: XNAS
search	[Optional] Use this parameter to search stock tickers by name or ticker symbol.
limit	[Optional] Specify a pagination limit (number of results per page) for your API request. Default limit value is 100, maximum allowed limit value is 1000.
offset	[Optional] Specify a pagination offset value for your API request. Example: An offset value of 100 combined with a limit value of 10 would show results 100-110. Default value is 0, starting with the first available result.

API Response:

{
    "pagination": {
        "limit": 100,
        "offset": 0,
        "count": 100,
        "total": 136785
    },
    "data": [
        {
            "name": "MICROSOFT CORP",
            "symbol": "MSFT",
            "stock_exchange": {
                "name": "NASDAQ Stock Exchange",
                "acronym": "NASDAQ",
                "mic": "XNAS",
                "country": "USA",
                "country_code": "US",
                "city": "New York",
                "website": "www.nasdaq.com",
            }
        },
        [...]
    ]
}

API Response Objects:

Response Object	Description
pagination > limit	Returns your pagination limit value.
pagination > offset	Returns your pagination offset value.
pagination > count	Returns the results count on the current page.
pagination > total	Returns the total count of results available.
name	Returns the name of the given stock ticker.
symbol	Returns the symbol of the given stock ticker.
stock_exchange > name	Returns the name of the stock exchange associated with the given stock ticker.
stock_exchange > acronym	Returns the acronym of the stock exchange associated with the given stock ticker.
stock_exchange > mic	Returns the MIC identification of the stock exchange associated with the given stock ticker.
stock_exchange > country	Returns the country of the stock exchange associated with the given stock ticker.
stock_exchange > country_code	Returns the 3-letter country code of the stock exchange associated with the given stock ticker.
stock_exchange > city	Returns the city of the stock exchange associated with the given stock ticker.
stock_exchange > website	Returns the website URL of the stock exchange associated with the given stock ticker.

Tickers:

JavaScript FetchJavaScript AxiosPython RequestsPython Http.client
                

const url = 'https://api.marketstack.com/v1/tickers?access_key={PASTE_YOUR_API_KEY_HERE}';
const options = {
	method: 'GET'
};

try {
	const response = await fetch(url, options);
	const result = await response.text();
	console.log(result);
} catch (error) {
	console.error(error);
}

                
            


ExchangesAvailable on: All plans
Using the exchanges API endpoint you will be able to look up information any of the 70+ stock exchanges supported by marketstack. You will be able to find and try out an example API request below.

Example API Request:

Run API Requesthttp://api.marketstack.com/v1/exchanges
    ? access_key = YOUR_MARKETSTACK_API_KEY

Endpoint Features:

Object	Description
/exchanges/[mic]	Obtain information about a specific stock exchange by attaching its MIC identification to your API request URL, e.g. /exchanges/XNAS.
/exchanges/[mic]/tickers	Obtain all available tickers for a specific exchange by attaching the exchange MIC as well as /tickers, e.g. /exchanges/XNAS/tickers.
/exchanges/[mic]/eod	Obtain end-of-day data for all available tickers from a specific exchange, e.g. /exchanges/XNAS/eod. For parameters, refer to End-of-day Data endpoint.
/exchanges/[mic]/intraday	Obtain intraday data for tickers from a specific exchange, e.g. /exchanges/XNAS/intraday. For parameters, refer to Intraday Data endpoint.
/exchanges/[mic]/eod/[date]	Obtain end-of-day data for a specific date in YYYY-MM-DD or ISO-8601 format. Example: /exchanges/XNAS/eod/2020-01-01.
/exchanges/[mic]/intraday/[date]	Obtain intraday data for a specific date and time in YYYY-MM-DD or ISO-8601 format. Example: /exchanges/IEXG/intraday/2020-05-21T00:00:00+0000.
/exchanges/[mic]/eod/latest	Obtain the latest end-of-day data for tickers of the given exchange. Example: /exchanges/XNAS/eod/latest
/exchanges/[mic]/intraday/latest	Obtain the latest intraday data for tickers of the given exchange. Example: /exchanges/IEXG/intraday/latest

Parameters:

Object	Description
access_key	[Required] Specify your API access key, available in your account dashboard.
search	[Optional] Use this parameter to search stock exchanges by name or MIC.
limit	[Optional] Specify a pagination limit (number of results per page) for your API request. Default limit value is 100, maximum allowed limit value is 1000.
offset	[Optional] Specify a pagination offset value for your API request. Example: An offset value of 100 combined with a limit value of 10 would show results 100-110. Default value is 0, starting with the first available result.

API Response:

{
    "pagination": {
        "limit": 100,
        "offset": 0,
        "count": 71,
        "total": 71
    },
    "data": [
        {
            "name": "NASDAQ Stock Exchange",
            "acronym": "NASDAQ",
            "mic": "XNAS",
            "country": "USA",
            "country_code": "US",
            "city": "New York",
            "website": "www.nasdaq.com",
            "timezone": {
                "timezone": "America/New_York",
                "abbr": "EST",
                "abbr_dst": "EDT"
            }
        },
        [...]
    ]
}

API Response Objects:

Response Object	Description
pagination > limit	Returns your pagination limit value.
pagination > offset	Returns your pagination offset value.
pagination > count	Returns the results count on the current page.
pagination > total	Returns the total count of results available.
name	Returns the name of the given stock exchange.
acronym	Returns the acronym of the given stock exchange.
mic	Returns the MIC identification of the given stock exchange.
country	Returns the country of the given stock exchange.
country_code	Returns the 3-letter country code of the given stock exchange.
city	Returns the given city of the stock exchange.
website	Returns the website URL of the given stock exchange.
timezone > timezone	Returns the timezone name of the given stock exchange.
timezone > abbr	Returns the timezone abbreviation of the given stock exchange.
timezone > abbr_dst	Returns the Summer time timezone abbreviation of the given stock exchange.

Exchanges:

JavaScript FetchJavaScript AxiosPython RequestsPython Http.client
                

const url = 'https://api.marketstack.com/v1/exchanges?access_key={PASTE_YOUR_API_KEY_HERE}';
const options = {
	method: 'GET'
};

try {
	const response = await fetch(url, options);
	const result = await response.text();
	console.log(result);
} catch (error) {
	console.error(error);
}

                
            


CurrenciesAvailable on: All plans
Using the currencies API endpoint you will be able to look up all currencies supported by the marketstack API. You will be able to find and try out an example API request below.

Example API Request:

Run API Requesthttp://api.marketstack.com/v1/currencies
    ? access_key = YOUR_MARKETSTACK_API_KEY

Parameters:

Object	Description
access_key	[Required] Specify your API access key, available in your account dashboard.
limit	[Optional] Specify a pagination limit (number of results per page) for your API request. Default limit value is 100, maximum allowed limit value is 1000.
offset	[Optional] Specify a pagination offset value for your API request. Example: An offset value of 100 combined with a limit value of 10 would show results 100-110. Default value is 0, starting with the first available result.

API Response:

{
    "pagination": {
        "limit": 100,
        "offset": 0,
        "count": 40,
        "total": 40
    },
    "data": [
        {
            "code": "USD",
            "name": "US Dollar",
            "symbol": "$",
            "symbol_native": "$",
        },
        [...]
    ]
}

API Response Objects:

Response Object	Description
pagination > limit	Returns your pagination limit value.
pagination > offset	Returns your pagination offset value.
pagination > count	Returns the results count on the current page.
pagination > total	Returns the total count of results available.
code	Returns the 3-letter code of the given currency.
name	Returns the name of the given currency.
symbol	Returns the text symbol of the given currency.
symbol_native	Returns the native text symbol of the given currency.

Currencies:

JavaScript FetchJavaScript AxiosPython RequestsPython Http.client
                

const url = 'https://api.marketstack.com/v1/currencies?access_key={PASTE_YOUR_API_KEY_HERE}';
const options = {
	method: 'GET'
};

try {
	const response = await fetch(url, options);
	const result = await response.text();
	console.log(result);
} catch (error) {
	console.error(error);
}

                
            


TimezonesAvailable on: All plans
Using the timezones API endpoint you will be able to look up information about all supported timezones. You will be able to find and try out an example API request below.

Example API Request:

Run API Requesthttp://api.marketstack.com/v1/timezones
    ? access_key = YOUR_MARKETSTACK_API_KEY

Parameters:

Object	Description
access_key	[Required] Specify your API access key, available in your account dashboard.
limit	[Optional] Specify a pagination limit (number of results per page) for your API request. Default limit value is 100, maximum allowed limit value is 1000.
offset	[Optional] Specify a pagination offset value for your API request. Example: An offset value of 100 combined with a limit value of 10 would show results 100-110. Default value is 0, starting with the first available result.

API Response:

{
    "pagination": {
        "limit": 100,
        "offset": 0,
        "count": 57,
        "total": 57
    },
    "data": [
        {
            "timezone": "America/New_York",
            "abbr": "EST",
            "abbr_dst": "EDT"
        },
        [...]
    ]
}

API Response Objects:

Response Object	Description
pagination > limit	Returns your pagination limit value.
pagination > offset	Returns your pagination offset value.
pagination > count	Returns the results count on the current page.
pagination > total	Returns the total count of results available.
timezone	Returns the name of the given timezone.
abbr	Returns the abbreviation of the given timezone.
abbr_dst	Returns the Summer time abbreviation of the given timezone.

Timezones:

JavaScript FetchJavaScript AxiosPython RequestsPython Http.client
                

const url = 'https://api.marketstack.com/v1/timezones?access_key={PASTE_YOUR_API_KEY_HERE}';
const options = {
	method: 'GET'
};

try {
	const response = await fetch(url, options);
	const result = await response.text();
	console.log(result);
} catch (error) {
	console.error(error);
}

                
            


FAQ
Ensuring our customers achieve success is paramount to what we do at APILayer. For this reason, we will be rolling out our Business Continuity plan guaranteeing your end users will never see a drop in coverage. Every plan has a certain amount of API calls that you can make in the given month. However, we would never want to cut your traffic or impact user experience negatively for your website or application in case you get more traffic.

What is an overage?
An overage occurs when you go over a quota for your API plan. When you reach your API calls limit, we will charge you a small amount for each new API call so we can make sure there will be no disruption in the service we provide to you and your website or application can continue running smoothly.

Prices for additional API calls will vary based on your plan. See table below for prices per call and example of an overage billing.

Plan Name	Monthly Price	Number of Calls	Overage Price per call	Overage	Total price
Basic	$9.99	10,000	0.001998	2,000	$13.99
Professional	$49.99	100,000	0.0009998	20,000	$69.99
Business	$149.99	500,000	0.00059996	100,000	$209.99
Why does APILayer have overage fees?
Overage fees allow developers to continue using an API once a quota limit is reached and give them time to upgrade their plan based on projected future use while ensuring API providers get paid for higher usage.

How do I know if I will be charged for overages?
When you are close to reaching your API calls limit for the month, you will receive an automatic notification (at 75%, 90% and 100% of your monthly quota). However, it is your responsibility to review and monitor for the plan’s usage limitations. You are required to keep track of your quota usage to prevent overages. You can do this by tracking the number of API calls you make and checking the dashboard for up-to-date usage statistics.

How will I be charged for my API subscription?
You will be charged for your monthly subscription plan, plus any overage fees applied. Your credit card will be billed after the billing period has ended.

What happens if I don’t have any overage fees?
In this case, there will be no change to your monthly invoice. Only billing cycles that incur overages will see any difference in monthly charges. The Business Continuity plan is an insurance plan to be used only if needed and guarantees your end users never see a drop in coverage from you.

What if I consistently have more API calls than my plan allows?
If your site consistently surpasses the set limits each month, you may face additional charges for the excess usage. Nevertheless, as your monthly usage reaches a certain threshold, it becomes more practical to consider upgrading to the next plan. By doing so, you ensure a smoother and more accommodating experience for your growing customer base.

I would like to upgrade my plan. How can I do that?
You can easily upgrade your plan by going to your Dashboard and selecting the new plan that would be more suitable for your business needs. Additionally, you may contact your Account Manager to discuss a custom plan if you expect a continuous increase in usage.


Introducing Platinum Support - Enterprise-grade support for APILayer
Upgrade your APIlayer subscription with our exclusive Platinum Support, an exceptional offering designed to enhance your business’ API management journey. With Platinum Support, you gain access to a host of premium features that take your support experience to a whole new level.

What does Platinum Support include?
Standard Support	Platinum Support
General review on the issue	Correct Icon	Correct Icon
Access to knowledge base articles	Correct Icon	Correct Icon
Email support communication	Correct Icon	Correct Icon
Regular products updates and fixes	Correct Icon	Correct Icon
Dedicated account team	Cross Icon	Correct Icon
Priority Email Support with unlimited communication	Cross Icon	Correct Icon
Priority bug and review updates	Cross Icon	Correct Icon
Option for quarterly briefing call with product Management	Cross Icon	Correct Icon
Features requests as priority roadmap input into product	Cross Icon	Correct Icon

Priority Email Support: Experience unrivaled responsiveness with our priority email support. Rest assured that your inquiries receive top-priority attention, ensuring swift resolutions to any issues.

Unlimited Communication: Communication is key, and with Platinum Support, you enjoy unlimited access to our support team. No matter how complex your challenges are, our experts are here to assist you every step of the way.

Priority Bug Review and Fixes: Bugs can be a headache, but not with Platinum Support. Benefit from accelerated bug review and fixes, minimizing disruptions and maximizing your API performance.

Dedicated Account Team: We understand the value of personalized attention. That's why Platinum Support grants you a dedicated account team, ready to cater to your specific needs and provide tailored solutions.

Quarterly Briefing Call with Product Team: Stay in the loop with the latest updates and insights from our Product team. Engage in a quarterly briefing call to discuss new features, enhancements, and upcoming developments.

Priority Roadmap Input: Your input matters! As a Platinum Support subscriber, your feature requests receive top priority, shaping our product roadmap to align with your evolving requirements.

Don't settle for the standard when you can experience the exceptional. Upgrade to Platinum Support today and supercharge your APIlayer experience!