# Cleaning report

| Step | Rows |
|---|---|
| Raw rows | 140,000 |
| Dropped: empty title or invalid citations | 0 |
| Dropped: placeholder titles (e.g. 'Article') | 1 |
| Dropped: exact duplicates (title, year, journal) | 31 |
| Kept, grouped: same title + year in different journals | 96 |
| Final rows | 139,968 |

## Label balance

| label_name   |   proportion |
|:-------------|-------------:|
| <60          |        0.296 |
| 60-120       |        0.393 |
| >120         |        0.311 |

## Label balance by year

|   year |   <60 |   60-120 |   >120 |
|-------:|------:|---------:|-------:|
|   2007 | 0     |    0.269 |  0.731 |
|   2008 | 0     |    0.434 |  0.566 |
|   2009 | 0     |    0.584 |  0.416 |
|   2010 | 0     |    0.735 |  0.265 |
|   2011 | 0.364 |    0.496 |  0.139 |
|   2012 | 0.758 |    0.193 |  0.049 |
|   2013 | 0.951 |    0.039 |  0.01  |

Unique journals after normalization: 5,075 (raw: 5,278)
