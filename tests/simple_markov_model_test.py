from ml_models.pgm.simple_markov_model import SimpleMarkovModel

train_data = [0, 1, 1, 1, 2, 2, 0, 4, 3, 2]
smm = SimpleMarkovModel(status_num=5)
smm.fit(train_data)
print(smm.predict_log_joint_prob([1, 2]))
print("\n")
print(smm.predict_prob_distribution(time_steps=2))
print("\n")
print(smm.predict_next_step_prob_distribution(current_status=2))
print("\n")
print(smm.predict_next_step_status(current_status=2))
print("\n")
print(smm.generate_status(search_type="beam"))
