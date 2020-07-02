/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "helper_functions.h"

using std::string;
using std::vector;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  /**
   * TODO: Set the number of particles. Initialize all particles to 
   *   first position (based on estimates of x, y, theta and their uncertainties
   *   from GPS) and all weights to 1. 
   * TODO: Add random Gaussian noise to each particle.
   * NOTE: Consult particle_filter.h for more information about this method 
   *   (and others in this file).
   */
   /* Set the number of particles */
  num_particles_ = 100  ;  
  /* Random number engine class that generates pseudo random numbers*/
  std::default_random_engine gen;
  /* Standard deviation values for x, y and theta*/
  double std_x{}, std_y{}, std_theta{};
  /* Create an object for the particle */
  Particle obj_part;

  /* create a noise (Gaussian noise) distribution for x, y and theta
  around mean 0 and standard deviation std_x, std_y and std_theta */
  std::normal_distribution<double> noise_x(0, std_x);
  std::normal_distribution<double> noise_y(0, std_y);
  std::normal_distribution<double> noise_theta(0, std_theta);
  
  /* If not initialized */
  while (!(initialized())) {
      /* Get standard deviation values for x, y and theta */
      std_x = std[0];
      std_y = std[1];
      std_theta = std[2];

      /* create a normal (Gaussian) distribution for x, y and theta 
      around mean x, y, and theta and standard deviation std_x, std_y and std_theta */
      std::normal_distribution<double> dist_x(x, std_x);
      std::normal_distribution<double> dist_y(y, std_y);
      std::normal_distribution<double> dist_theta(theta, std_theta);

      /* Allocate memory to the vector */
      particles_.reserve(num_particles_);

      /* Get the values for the Particle structure */
      for (int i = 0; i < num_particles_; i++) {
          obj_part.id = i;
          obj_part.x = dist_x(gen) + noise_x(gen);
          obj_part.y = dist_y(gen) + noise_y(gen);
          obj_part.theta = dist_theta(gen) + noise_theta(gen);
          obj_part.weight = 1.0;
          particles_.push_back(obj_part);
          weights_.push_back(1.0);
      }

      std::cout << weights_.size() << std::endl;
      
      /* Initialize to true after initializing the particles */
      is_initialized_ = true;
  }
}

void ParticleFilter::prediction(double delta_t, double std_pos[],
    double velocity, double yaw_rate) {
    /**
     * TODO: Add measurements to each particle and add random Gaussian noise.
     * NOTE: When adding noise you may find std::normal_distribution
     *   and std::default_random_engine useful.
     *  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
     *  http://www.cplusplus.com/reference/random/default_random_engine/
     */
    std::cout << "prediction" << std::endl;

    /* Random number engine class that generates pseudo random numbers */
    std::default_random_engine gen;
    /* Standard deviation values for x, y and theta*/
    double std_x{}, std_y{}, std_theta{};

    /* Get standard deviation values for x, y and theta */
    std_x = std_pos[0];
    std_y = std_pos[1];
    std_theta = std_pos[2];

    /* Create a noise (Gaussian noise) distribution for x, y and theta
    around mean 0 and standard deviation std_x, std_y and std_theta */
    std::normal_distribution<double> dist_x(0, std_x);
    std::normal_distribution<double> dist_y(0, std_y);
    std::normal_distribution<double> dist_theta(0, std_theta);

    /* Every particle is moved at certain distance at a certain heading after delta t */
    for_each (particles_.begin(), particles_.end(), [&](Particle particle)
        {
            particle.weight = 1;

            if (fabs(yaw_rate) < 0.001) {
                particle.x += velocity * delta_t * cos(particle.theta);
                particle.y += velocity * delta_t * sin(particle.theta);
            }
            else {
                particle.x += velocity / yaw_rate * (sin(particle.theta + yaw_rate * delta_t) - sin(particle.theta));
                particle.y += velocity / yaw_rate * (cos(particle.theta) - cos(particle.theta + yaw_rate * delta_t));
                particle.theta += yaw_rate * delta_t;
            }

            /* Add noise to every particle after upating it with motion */
            particle.x += dist_x(gen);
            particle.y += dist_y(gen);
            particle.theta += dist_theta(gen);
        });
}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted, 
                                     vector<LandmarkObs>& observations) {
  /**
   * TODO: Find the predicted measurement that is closest to each 
   *   observed measurement and assign the observed measurement to this 
   *   particular landmark.
   * NOTE: this method will NOT be called by the grading code. But you will 
   *   probably find it useful to implement this method and use it as a helper 
   *   during the updateWeights phase.
   */

    /* Define variables for finding nearest neighbour assigning nearest distance */
    double nearest_neighbour{}, minimum_distance{};
    /* Variable to define landmark id */
    int landmark_id{};
    
    /*For all the observated landmarks or the sensor measurements */
    for (auto& obs_meas : observations) {
        
        /* Take minimum distance as the maximum for the initial comparison*/
        minimum_distance = std::numeric_limits<double>::max();

        /* Set initial nearest particle id to -1 to ensure that the predicted measurement was 
        mapped to the observed one */
        obs_meas.id = -1;

        /*For all the predicted landmarks or the map landmarks */
        for (const auto& pred_meas : predicted) {
            
            /*the nearest neighbour is calculated by finding eucledian distance between 
            predicted and obsereved points  */
            nearest_neighbour = dist(pred_meas.x, pred_meas.y, obs_meas.x, obs_meas.y);
             
            /* Find the nearest neighbour by finding the minimum distance between 
            predicted and obsereved landmark */
            if (nearest_neighbour < minimum_distance) {
                minimum_distance = nearest_neighbour;
                /* Assign predicted measurement id to a variable*/
                landmark_id = pred_meas.id;
            }
        }

        /* Assign that measured landmark id to the observed id who is the
        nearest neighbour */
        obs_meas.id = landmark_id;
    }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
                                   const vector<LandmarkObs> &observations, 
                                   const Map &map_landmarks) {
  /**
   * TODO: Update the weights of each particle using a mult-variate Gaussian 
   *   distribution. You can read more about this distribution here: 
   *   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
   * NOTE: The observations are given in the VEHICLE'S coordinate system. 
   *   Your particles are located according to the MAP'S coordinate system. 
   *   You will need to transform between the two systems. Keep in mind that
   *   this transformation requires both rotation AND translation (but no scaling).
   *   The following is a good resource for the theory:
   *   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
   *   and the following is a good resource for the actual equation to implement
   *   (look at equation 3.33) http://planning.cs.uiuc.edu/node99.html
   */

    /* First Step: Transform car sensor landmark observation from the car co-ordinate system to map
    co-ordinate system for every particle */  

    /* Initialize particle x, y co-ordinates, and its sine and cos theta */
    double x_p{}, y_p{}, sin_theta{}, cos_theta{};
    /* Define distance between particle and the map landmark */
    double distance{};

    /* Loop through every particle */
    for (auto& particle : particles_) { 
        /* Initialize vector to store the transormed co-ordinates */
        std::vector<LandmarkObs> transformed_cordinates{};
        /* Initialize vector to store the global co-ordinates */
        std::vector<LandmarkObs> global_cordinates{};
        /* Initialize the observed measurements for every particle */
        double x_c{}, y_c{}, obs_id;  

        /* Set the size for the transformed co-ordinates */
        transformed_cordinates.reserve(observations.size());

        /* Get the x and y co-ordinates of the particle */
        x_p = particle.x;
        y_p = particle.y;

        /* Get the sine and cos of the particle */
        sin_theta = sin(particle.theta);
        cos_theta = cos(particle.theta);

/////////////////////////////////////////////////////////////////////////////////////////////////////////////
        /* Store the transformed co-ordinates */
        std::transform(observations.begin(), observations.end(), std::back_inserter(transformed_cordinates),
            [&](LandmarkObs obs_meas) {
                /* Get the x and y co-ordinates of the observed measurements */
                x_c = obs_meas.x;
                y_c = obs_meas.y;
                obs_id = obs_meas.id;

                /* Intialize the object of the structure LandmarkObs to store values */
                LandmarkObs trans_cord{};

                /* Update the structure */
                trans_cord.x = x_p + cos_theta * x_c - sin_theta * y_c;
                trans_cord.y = y_p + sin_theta * x_c + cos_theta * y_c;
                trans_cord.id = obs_id;

                return trans_cord;
            });

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

        /* Second Step: Associating these transformed observtions with the nearest landmark on the map */        
        for (const auto& glob_cord : map_landmarks.landmark_list) {
            /* Define structure for landmark */
            LandmarkObs map{};

            /* Check whether the distance between the particle and the map landmark is within the sensor range */
            distance = dist(x_p, y_p, glob_cord.x_f, glob_cord.y_f);           

            /* Update landmark structure if distance is within the sensor range */
            if (distance < sensor_range) {   

                map.x = glob_cord.x_f;
                map.y = glob_cord.y_f;
                map.id = glob_cord.id_i;

                /* Update the global co-ordinate structure */
                global_cordinates.push_back(map);
            }
        }        

        /* Associate these transformed observtions with the nearest landmark on the map */
        /* Here global_cordinates is the prediction and the transformed cordinate is the observation */
        dataAssociation(global_cordinates, transformed_cordinates);

        /* Third Step: Update particle weight */        

        /* Set values for multi-variate Gaussian distribution */
        auto cov_x = std_landmark[0] * std_landmark[0];
        auto cov_y = std_landmark[1] * std_landmark[1];
        auto normalizer = 2.0 * M_PI * std_landmark[0] * std_landmark[1];

        /* Check if the transformed cordinates id gets matched with global cordinates id */
        for (int i = 0; i < transformed_cordinates.size(); i++) {
            for (int j = 0; j < global_cordinates.size(); j++) {
                if (transformed_cordinates[i].id == global_cordinates[j].id) {
                    auto diff_x = transformed_cordinates[i].x - global_cordinates[j].x;
                    auto diff_y = transformed_cordinates[i].y - global_cordinates[j].y;

                    std::cout << particle.weight << std::endl;
                    std::cout << diff_x << diff_y << cov_x << cov_y << normalizer << std::endl;
                    particle.weight *= exp(-(diff_x * diff_x / (2 * cov_x) + diff_y * diff_y / (2 * cov_y))) / normalizer;                    
                }
            }
        }         
    } 

    /* Update the weights vector */
    for (int idx = 0; idx < weights_.size(); idx++) {
        weights_[idx] = particles_[idx].weight;
        std::cout << weights_[idx] << std::endl;
    }
 }

void ParticleFilter::resample() {
  /**
   * TODO: Resample particles with replacement with probability proportional 
   *   to their weight. 
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */
    /* Create a vector of new particles */
    std::vector<Particle> new_particles(num_particles_);
    /* Random number engine class that generates pseudo random numbers */
    std::default_random_engine gen;
    /*std::discrete_distribution produces random integers on the interval [0, n), where the probability 
    of each individual integer i is defined as w i/S, that is the weight of the ith integer divided by the sum of all n weights.*/
    std::discrete_distribution<size_t> distr_index(weights_.begin(), weights_.end());

    /* Create new particles with probability proportional to their weight */
    for (auto i = 0; i < particles_.size(); i++) {
        new_particles[i] = particles_[distr_index(gen)];
    }

    /* Copy it to the original particle vector */
    particles_ = std::move(new_particles);
}

void ParticleFilter::SetAssociations(Particle& particle, 
                                     const vector<int>& associations, 
                                     const vector<double>& sense_x, 
                                     const vector<double>& sense_y) {
  // particle: the particle to which assign each listed association, 
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}