# Performance speed between Jurassic, ClaudeV2 and Cohere with Bedrock

top_p = 1
temperature = 0
top_k = 500

| Prompt Type                                  |   Jurassic Ultra |   ClaudeV2 |   Cohere Text V14 |
|----------------------------------------------|------------------|------------|-------------------|
| simple_prompt                                |         1.65748  |  19.4163   |          9.91002  |
| meeting_transcribe_prompt                    |         1.57179  |   4.76434  |          7.27064  |
| article_summarisation_prompt                 |         2.43853  |   5.95223  |          7.24761  |
| content_generation_prompt                    |         1.37147  |   3.90778  |          3.13677  |
| create_a_table_of_product_description_prompt |         3.84413  |  20.2166   |         16.9797   |
| extract_topics_and_sentiment_from_reviews    |        23.2477   |   3.23922  |          2.07059  |
| generate_product_descriptions_prompt         |         1.08525  |   7.64143  |          4.16144  |
| information_extraction_prompt                |         0.892964 |   1.48579  |          1.3171   |
| multiple_choice_classification               |         0.519957 |   7.12836  |          3.1104   |
| outline_generation_prompt                    |         8.61899  |  11.1042   |          5.82898  |
| question_and_answer_prompt                   |         0.376151 |   0.635324 |          0.373321 |
| remove_pii_prompt                            |         1.85442  |   3.01947  |          3.5111   |
| summarise_the_key_takeaways                  |         2.38159  |   4.11257  |          5.49059  |
| write_an_article                             |         2.50646  |  10.6105   |          9.24618  |
| write_a_promo_doc                            |         3.72872  |  15.2176   |          9.536    |
| code_generation_prompt                       |         2.56087  |  12.0985   |         14.8708   |
| receipe_generation_prompt                    |         3.65228  |  14.7581   |         12.8457   |
| zero_shot_prompt                             |         0.844149 |  20.7179   |          5.03868  |
| few_shot_prompt                              |         0.441008 |   5.81665  |          2.90261  |



NOTE: This performance guide is for reference only. 