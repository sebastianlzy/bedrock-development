# Performance speed between Jurassic, ClaudeV2 and Cohere with Bedrock


| Prompt Type                                  |   Jurassic |   ClaudeV2 |    Cohere |
|----------------------------------------------|------------|------------|-----------|
| simple_prompt                                |   2.15445  |  12.9554   | 10.2942   |
| meeting_transcribe_prompt                    |   1.4663   |   4.07018  | 10.0461   |
| article_summarisation_prompt                 |   2.91254  |   6.22098  |  7.22955  |
| content_generation_prompt                    |   1.14321  |   4.09192  |  4.53423  |
| create_a_table_of_product_description_prompt |   3.43854  |   7.90823  | 14.9399   |
| extract_topics_and_sentiment_from_reviews    |   1.21884  |   4.1012   |  2.14875  |
| generate_product_descriptions_prompt         |   1.1275   |   6.86707  |  4.08269  |
| information_extraction_prompt                |   0.867989 |   1.18197  |  1.29382  |
| multiple_choice_classification               |   0.485759 |  10.7294   |  3.6281   |
| outline_generation_prompt                    |   3.70445  |  10.7805   |  6.81991  |
| question_and_answer_prompt                   |   0.362269 |   0.758142 |  0.362061 |
| remove_pii_prompt                            |   2.06959  |   3.42438  |  5.10182  |
| summarise_the_key_takeaways                  |   1.89568  |   7.76748  | 10.1706   |
| write_an_article                             |   2.28856  |   9.33375  |  9.02295  |
| write_a_promo_doc                            |   3.4125   |  17.8488   | 11.5951   |
| code_generation_prompt                       |   2.69715  |   6.15809  |  5.88088  |


NOTE: This performance guide is for reference only. 